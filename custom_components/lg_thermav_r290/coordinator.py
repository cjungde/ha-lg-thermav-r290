"""DataUpdateCoordinator for the LG ThermaV R290."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_BAUDRATE,
    CONF_CONNECTION_TYPE,
    CONF_PARITY,
    CONF_SCAN_INTERVAL,
    CONF_SERIAL_PORT,
    CONF_SLAVE_ID,
    CONF_STOPBITS,
    CONNECTION_TCP,
    CONNECTION_TCP_RTU,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

_MODBUS_TIMEOUT = 10


def _detect_slave_kwarg() -> str | None:
    """Return the keyword pymodbus uses for the slave/device ID (version-dependent).

    pymodbus < 3.5  : "unit"
    pymodbus 3.5-3.8: "slave"
    pymodbus 3.9+   : "device_id"  (keyword-only, no **kwargs)
    Returns None if the method accepts no per-request ID at all.
    """
    import inspect

    try:
        from pymodbus.client import AsyncModbusTcpClient

        params = inspect.signature(AsyncModbusTcpClient.read_input_registers).parameters
        for name in ("slave", "unit", "device_id"):
            if name in params:
                return name
        # **kwargs present — older versions absorb slave this way
        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return "slave"
        return None
    except Exception:
        return "slave"


_SLAVE_KWARG: str | None = _detect_slave_kwarg()


def _build_client(data: dict) -> Any:
    """Instantiate the appropriate pymodbus async client from config entry data."""
    from pymodbus.client import AsyncModbusSerialClient, AsyncModbusTcpClient

    conn_type = data.get(CONF_CONNECTION_TYPE, CONNECTION_TCP)

    if conn_type == CONNECTION_TCP:
        return AsyncModbusTcpClient(
            host=data["host"],
            port=data.get("port", 502),
            timeout=_MODBUS_TIMEOUT,
            retries=1,
        )

    if conn_type == CONNECTION_TCP_RTU:
        try:
            from pymodbus.framer import FramerType

            framer = FramerType.RTU
        except ImportError:
            framer = "rtu"
        return AsyncModbusTcpClient(
            host=data["host"],
            port=data.get("port", 502),
            framer=framer,
            timeout=_MODBUS_TIMEOUT,
            retries=1,
        )

    try:
        from pymodbus.framer import FramerType

        framer = FramerType.RTU
    except ImportError:
        framer = "rtu"

    return AsyncModbusSerialClient(
        port=data[CONF_SERIAL_PORT],
        framer=framer,
        baudrate=data.get(CONF_BAUDRATE, 9600),
        parity=data.get(CONF_PARITY, "N"),
        stopbits=data.get(CONF_STOPBITS, 1),
        bytesize=8,
        timeout=_MODBUS_TIMEOUT,
        retries=1,
    )


def _signed16(value: int) -> int:
    """Convert an unsigned 16-bit register value to a signed integer."""
    return value - 65536 if value >= 32768 else value


class LGThermaVCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Polls the LG ThermaV R290 at a fixed interval and distributes data to entities."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._entry = entry
        self._client: Any = None
        self._modbus_lock = asyncio.Lock()
        scan_interval = entry.options.get(
            CONF_SCAN_INTERVAL,
            entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        )
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=scan_interval),
        )

    async def _get_client(self) -> Any:
        """Return a connected Modbus client, reconnecting if stale."""
        if self._client is not None:
            if self._client.connected:
                return self._client
            self._client.close()
            self._client = None

        client = _build_client(self._entry.data)
        try:
            connected = await asyncio.wait_for(
                client.connect(), timeout=_MODBUS_TIMEOUT
            )
        except TimeoutError as exc:
            client.close()
            raise UpdateFailed("Connection to LG ThermaV timed out") from exc
        if not connected:
            client.close()
            raise UpdateFailed("Could not connect to LG ThermaV")
        self._client = client
        return self._client

    async def _async_update_data(self) -> dict[str, Any]:
        from pymodbus.exceptions import ModbusException

        slave = int(self._entry.data[CONF_SLAVE_ID])

        async with self._modbus_lock:
            try:
                client = await self._get_client()

                def _slave_kwargs(**extra) -> dict:
                    kw = dict(extra)
                    if _SLAVE_KWARG:
                        kw[_SLAVE_KWARG] = slave
                    return kw

                async def read_input(address: int, count: int) -> list[int]:
                    result = await client.read_input_registers(
                        address, **_slave_kwargs(count=count)
                    )
                    if hasattr(result, "isError") and result.isError():
                        raise UpdateFailed(f"Modbus FC04 error at {address}: {result}")
                    return result.registers

                async def read_holding(address: int, count: int) -> list[int]:
                    result = await client.read_holding_registers(
                        address, **_slave_kwargs(count=count)
                    )
                    if hasattr(result, "isError") and result.isError():
                        raise UpdateFailed(f"Modbus FC03 error at {address}: {result}")
                    return result.registers

                async def read_coils(address: int, count: int) -> list[bool]:
                    result = await client.read_coils(
                        address, **_slave_kwargs(count=count)
                    )
                    if hasattr(result, "isError") and result.isError():
                        raise UpdateFailed(f"Modbus FC01 error at {address}: {result}")
                    return result.bits[:count]

                async def read_discrete(address: int, count: int) -> list[bool]:
                    result = await client.read_discrete_inputs(
                        address, **_slave_kwargs(count=count)
                    )
                    if hasattr(result, "isError") and result.isError():
                        raise UpdateFailed(f"Modbus FC02 error at {address}: {result}")
                    return result.bits[:count]

                # Read all blocks
                # Register 15 is undefined on this device; split the input read to avoid it.
                inp_lo = await read_input(0, 14)  # addresses 0-13
                inp_hi = await read_input(16, 9)  # addresses 16-24 (skips 14-15)
                hold_low = await read_holding(0, 10)  # addresses 0-9
                coils = await read_coils(0, 6)  # addresses 0-5
                di = await read_discrete(0, 17)  # addresses 0-16

                # Optional product info registers — not available on all firmware versions
                try:
                    inp_product_group = await read_input(9997, 2)
                    device_group: int | None = inp_product_group[0]
                    product_info: int | None = inp_product_group[1]
                except UpdateFailed:
                    device_group = None
                    product_info = None

                return {
                    # Input registers (raw × scale)
                    # inp_lo[n] = register n  (0-13)
                    # inp_hi[n] = register 16+n (16-24)
                    "error_code": inp_lo[0],
                    "odu_operation_cycle": inp_lo[1],
                    "inlet_temp": round(inp_lo[2] * 0.1, 1),
                    "outlet_temp": round(inp_lo[3] * 0.1, 1),
                    "backup_heater_outlet_temp": round(inp_lo[4] * 0.1, 1),
                    "dhw_water_temp": round(inp_lo[5] * 0.1, 1),
                    "solar_collector_temp": round(inp_lo[6] * 0.1, 1),
                    "room_air_temp_circuit1": round(inp_lo[7] * 0.1, 1),
                    "flow_rate": round(inp_lo[8] * 0.1, 1),
                    "flow_temp_circle_2": round(inp_lo[9] * 0.1, 1),
                    "room_air_temp_circuit2_in": round(inp_lo[10] * 0.1, 1),
                    "energy_state_input": inp_lo[11],
                    "outside_temp": round(inp_lo[12] * 0.1, 1),
                    "water_pressure": round(inp_lo[13] * 0.1, 1),
                    "temp_liquid_gas": round(inp_hi[0] * 0.1, 1),
                    "temp_suction": round(inp_hi[2] * 0.1, 1),
                    "temp_heatgas": round(inp_hi[3] * 0.1, 1),
                    "temp_before_vaporiser": round(inp_hi[4] * 0.1, 1),
                    "temp_after_vaporiser": round(inp_hi[5] * 0.1, 1),
                    "high_pressure": round(inp_hi[6] * 0.1, 1),
                    "low_pressure": round(inp_hi[7] * 0.1, 1),
                    "compressor_rpm": int(round(inp_hi[8] * 60)),
                    "device_group": device_group,
                    "product_info": product_info,
                    # Holding registers
                    "operation_mode": hold_low[0],
                    "control_method": hold_low[1],
                    "target_temp_circuit1": round(hold_low[2] * 0.1, 1),
                    "room_air_temp_circuit1_hold": round(hold_low[3] * 0.1, 1),
                    "shift_value_circuit1": _signed16(hold_low[4]),
                    "target_temp_circuit2": round(hold_low[5] * 0.1, 1),
                    "room_air_temp_circuit2_hold": round(hold_low[6] * 0.1, 1),
                    "shift_value_circuit2": _signed16(hold_low[7]),
                    "dhw_target_temp": round(hold_low[8] * 0.1, 1),
                    "energy_state_raw": hold_low[9],
                    # Coils
                    "coil_hauptschalter": coils[0],
                    "coil_dhw": coils[1],
                    "coil_silent_mode": coils[2],
                    "coil_dhw_desinfection": coils[3],
                    "coil_emergency_stop": coils[4],
                    "coil_emergency_trigger": coils[5],
                    # Discrete inputs
                    # Discrete input 10001. The manual documents this as
                    # "0 = flow rate OK / 1 = flow rate too low", but on the R290
                    # monoblock it reports the opposite: 1 means flow is present.
                    # Measured 2026-08-18, four transitions in one session, di[0]
                    # and di[1] (pump) identical in every sample of the same poll:
                    #   flow  5.0 LPM, pump off -> di[0]=0, di[0:6]=[0,0,0,0,0,0]
                    #   flow 15.3 LPM, pump on  -> di[0]=1, di[0:6]=[1,1,0,0,0,0]
                    #   flow 29.5 LPM, pump on  -> di[0]=1, di[0:6]=[1,1,0,0,0,1]
                    # Reporting "rate too low" at 29.5 LPM makes no sense, and a
                    # read offset is ruled out because di[3] (compressor) and
                    # di[5] (DHW heating) stay correctly aligned. Exposed as a
                    # RUNNING sensor rather than a PROBLEM one, which previously
                    # raised a false alert on every pump run.
                    "di_water_flow": di[0],  # 1 = flow present
                    "di_water_pump": di[1],
                    "di_ext_water_pump": di[2],
                    "di_compressor": di[3],
                    "di_defrosting": di[4],
                    "di_dhw_heating": di[5],
                    "di_dhw_tank_desinfection": di[6],
                    "di_silent_mode": di[7],
                    "di_cooling": di[8],
                    "di_solar_pump": di[9],
                    "di_backup_heater_step1": di[10],
                    "di_backup_heater_step2": di[11],
                    "di_dhw_boost_heater": di[12],
                    "di_error": di[13],
                    "di_emergency_heating_cooling": di[14],
                    "di_emergency_dhw": di[15],
                    "di_mix_pump": di[16],
                }

            except ModbusException as exc:
                self._client = None
                raise UpdateFailed(f"Modbus communication error: {exc}") from exc
            except UpdateFailed:
                raise
            except Exception as exc:
                self._client = None
                raise UpdateFailed(
                    f"Unexpected error polling LG ThermaV: {exc}"
                ) from exc

    async def async_write_coil(self, address: int, value: bool) -> None:
        """Write a single coil register."""
        from pymodbus.exceptions import ModbusException

        slave = int(self._entry.data[CONF_SLAVE_ID])
        async with self._modbus_lock:
            try:
                client = await self._get_client()
                kwargs = {_SLAVE_KWARG: slave} if _SLAVE_KWARG else {}
                result = await client.write_coil(address, value, **kwargs)
                if hasattr(result, "isError") and result.isError():
                    _LOGGER.error("Write coil %d failed: %s", address, result)
            except ModbusException as exc:
                self._client = None
                _LOGGER.error("Modbus error writing coil %d: %s", address, exc)

    async def async_write_register(self, address: int, value: int) -> None:
        """Write a single holding register (FC06)."""
        from pymodbus.exceptions import ModbusException

        slave = int(self._entry.data[CONF_SLAVE_ID])
        async with self._modbus_lock:
            try:
                client = await self._get_client()
                kwargs = {_SLAVE_KWARG: slave} if _SLAVE_KWARG else {}
                result = await client.write_register(address, value, **kwargs)
                if hasattr(result, "isError") and result.isError():
                    _LOGGER.error("Write register %d failed: %s", address, result)
            except ModbusException as exc:
                self._client = None
                _LOGGER.error("Modbus error writing register %d: %s", address, exc)

    async def async_close(self) -> None:
        """Close the Modbus connection on integration unload."""
        if self._client is not None:
            self._client.close()
            self._client = None
