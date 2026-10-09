/*
  FB9 joystick firmware (Arduino Uno or Nano).

  What it does:
    The laptop sends ONE byte per update over USB serial at 115200 baud.
    The byte is a bitmask of which joystick buttons should be pressed:
        bit 0 = UP     (value 1)
        bit 1 = DOWN   (value 2)
        bit 2 = LEFT   (value 4)
        bit 3 = RIGHT  (value 8)
        bit 4 = FIRE   (value 16)
    Example: 21 = 1 + 4 + 16 = UP + LEFT + FIRE pressed (binary 10101).

  Which Arduino pin drives which joystick button:
    D2 -> UP        D3 -> DOWN      D4 -> LEFT      D5 -> RIGHT      D6 -> FIRE
    (each pin goes to the IN side of one optocoupler channel, see docs/plan section 5)

  ACTIVE_LOW:
    Many PC817 optocoupler modules are "active LOW": they switch ON when their
    input pin is pulled LOW. Then "pressed" means the pin is LOW.
    Set ACTIVE_LOW to 1 for those modules. Set it to 0 if your module switches on
    with HIGH. Check with a multimeter BEFORE plugging into the console: the output
    should be closed (continuity to G2) only while the pin is in its "pressed" state.

  Safety failsafe:
    If no byte arrives for 200 milliseconds (laptop crashed, cable pulled, ...),
    every button is released. The laptop should keep sending its state
    (it does, roughly 15 to 60 times per second).

  Contradictions:
    UP + DOWN together, or LEFT + RIGHT together, are not possible on a real stick,
    so both are released.

  The Arduino never sends anything back (fire-and-forget).
*/

#define ACTIVE_LOW 1          // 1 = pressed means pin LOW (typical PC817 module); 0 = pressed means pin HIGH
#define BAUD 115200
#define FAILSAFE_MS 200UL     // release everything after this much silence

const uint8_t PIN_UP = 2;
const uint8_t PIN_DOWN = 3;
const uint8_t PIN_LEFT = 4;
const uint8_t PIN_RIGHT = 5;
const uint8_t PIN_FIRE = 6;

const uint8_t BIT_UP = 1;     // bit 0
const uint8_t BIT_DOWN = 2;   // bit 1
const uint8_t BIT_LEFT = 4;   // bit 2
const uint8_t BIT_RIGHT = 8;  // bit 3
const uint8_t BIT_FIRE = 16;  // bit 4

// Level that means "button pressed" and "button released" for the optocouplers.
const uint8_t LEVEL_PRESSED = ACTIVE_LOW ? LOW : HIGH;
const uint8_t LEVEL_RELEASED = ACTIVE_LOW ? HIGH : LOW;

uint8_t currentMask = 0;           // the last valid mask we received (0..31)
unsigned long lastByteMs = 0;      // millis() time of the last byte received

// Drive one pin to the pressed or released level.
void setButton(uint8_t pin, bool pressed) {
  digitalWrite(pin, pressed ? LEVEL_PRESSED : LEVEL_RELEASED);
}

// Make the outputs match a bitmask. Contradictory pairs release both buttons.
void applyMask(uint8_t mask) {
  bool up = (mask & BIT_UP) != 0;
  bool down = (mask & BIT_DOWN) != 0;
  bool left = (mask & BIT_LEFT) != 0;
  bool right = (mask & BIT_RIGHT) != 0;
  bool fire = (mask & BIT_FIRE) != 0;

  if (up && down) {
    up = false;
    down = false;
  }
  if (left && right) {
    left = false;
    right = false;
  }

  setButton(PIN_UP, up);
  setButton(PIN_DOWN, down);
  setButton(PIN_LEFT, left);
  setButton(PIN_RIGHT, right);
  setButton(PIN_FIRE, fire);
}

void setup() {
  // Write the "released" level BEFORE switching the pins to outputs: pinMode(OUTPUT) alone
  // starts the pin LOW, which an ACTIVE_LOW module would read as a press.
  applyMask(0);
  pinMode(PIN_UP, OUTPUT);
  pinMode(PIN_DOWN, OUTPUT);
  pinMode(PIN_LEFT, OUTPUT);
  pinMode(PIN_RIGHT, OUTPUT);
  pinMode(PIN_FIRE, OUTPUT);

  Serial.begin(BAUD);
  lastByteMs = millis();
}

void loop() {
  // Read every byte that has arrived; only the newest one matters.
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    if (b <= 31) {                 // ignore bytes that are not a valid mask
      currentMask = b;
    }
    lastByteMs = millis();
  }

  // Failsafe: no byte for 200 ms means release everything.
  // (millis() subtraction stays correct even when millis() wraps around.)
  if (millis() - lastByteMs > FAILSAFE_MS) {
    currentMask = 0;
  }

  applyMask(currentMask);
}
