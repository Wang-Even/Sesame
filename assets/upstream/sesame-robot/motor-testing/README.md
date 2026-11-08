## How to test the motors on the DIY Lolin S2 mini build

> [!TIP]
> This won't work if you don't have the motor testing firmware installed from the `/sesame-robot/firmware/debugging-firmware/sesame-motor-tester.ino`

### Step 1: Basic Motor Testing

> [!CAUTION]
> Don't attach the hips while testing. All servos should be free to move without anything blocking them.

1. Go to Arduino IDE and open the serial monitor.
![Serial monitor](how%20to%20open%20searial%20monitor%20arudinoide.png)
2. Plug motor 0 into the correct port.
3. In the serial monitor, type `0,90`.
4. Type `stop` into the serial monitor to stop all servos moving.
5. If the motor moves correctly, unplug it and change to the next motor.
6. Repeat this process for all remaining servos by entering `numberofservo,90`.

---

### Step 2: Hip and Leg Joint Calibration

Once all individual motors are tested and working, follow these steps to align the legs properly.

1. **Remove the hip joints** from the servo splines.
2. **Command REST** via the serial monitor. 
   * All four hip servos will now move to 90°.
3. While the motors are holding at 90°, **fit each hip joint** so it sits **PARALLEL** to the side/body of the robot.
4. **Do not screw them down yet.**
5. **Command STAND** via the serial monitor.
   * **R1** moves 45° one way.
   * **R2** moves 45° the other way.
   * **L1** moves 45° the other way.
   * **L2** moves 45° one way.
   * *The legs should now form an "X" shape.*
6. **Check the stance** to ensure the four legs form the intended mirrored "X" shape.
7. **Go back to REST**.
   * All four legs should return to being perpendicular to the body.
8. **Secure the joints** by installing the centre screws only after verifying the alignment is correct.
