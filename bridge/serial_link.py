"""Serial link to the Arduino joystick firmware (bridge/arduino/joystick.ino): one byte = button bitmask."""
import time

import serial

MAX_MASK = 0b11111  # bit0 UP, bit1 DOWN, bit2 LEFT, bit3 RIGHT, bit4 FIRE


def _check(mask: int) -> int:
    if not 0 <= int(mask) <= MAX_MASK:
        raise ValueError(f"mask must be 0..{MAX_MASK}, got {mask}")
    return int(mask)


class JoystickLink:
    """Real link. Opening the port resets an Uno (DTR), so we wait `settle_s` before the first byte."""

    def __init__(self, port: str, baud: int = 115200, settle_s: float = 2.0):
        self.ser = serial.Serial(port, baud, timeout=0, write_timeout=1.0)
        time.sleep(settle_s)
        self.ser.reset_input_buffer()
        self.last = 0

    def send(self, mask: int) -> None:
        """Send the current button state. Re-send at least every 200 ms (the firmware releases after 200 ms of silence)."""
        self.ser.write(bytes([_check(mask)]))
        self.last = int(mask)

    def release(self) -> None:
        self.send(0)

    def close(self) -> None:
        try:
            self.release()
        finally:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeLink:
    """Test double: records every mask sent instead of writing to a serial port."""

    def __init__(self):
        self.masks: list[int] = []
        self.last = 0

    def send(self, mask: int) -> None:
        self.last = _check(mask)
        self.masks.append(self.last)

    def release(self) -> None:
        self.send(0)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
