# Eye-tracking lab (M11)

Checks the model against what real people look at. Open the app, choose **Eye lab**.

| Tab | Needs | What happens |
|---|---|---|
| **Live tracker** | webcam | A glowing blob follows your gaze across the whole screen with a short trail. Use it to see how good the tracking is and to check it against the rings on screen. |
| **Live demo** | webcam | Look at one thumbnail for 3 s, then see the prediction next to your real gaze (CC, NSS, "was the first look predicted?"). |
| **Webcam test** | webcam | Same, over several thumbnails in random order (the pilot protocol). |
| **Click test** | nothing | Thumbnail flashes for 1.5 s and disappears; the viewer clicks where they looked first, then second. |

The **Input** menu offers *Webcam*, *Mouse stand-in* (the pointer acts as the eyes, for rehearsal) and, in dev builds only, *Simulated webcam* (a fake tracker with a deliberate error, used to test the calibration flow).

## Calibration: asked once, remembered

- The **first** run does a full calibration (about a minute): 13 dots. Look at a dot, click it once, keep looking until it fills; the tracker learns from ~16 samples per dot. Then a 5-point accuracy check.
- The result is **saved in this browser only** (localStorage, ~0.4 MB; never sent anywhere). After that you are not asked again:
  - same visit, checked in the last 15 min: nothing at all;
  - a new visit: a **5-point quick check** (~8 s) that also re-fits the correction, so small changes in how you sit are absorbed;
  - a full calibration again only if the screen/window size changed (>3 %), the saved one is over 30 days old, or you press **Forget it** / **Recalibrate**.
- A calibration that is **poor** (worse than 9 % of the screen width) is used for that visit but *not saved*; one worse than 15 % is rejected outright. The tracker bar always shows the saved accuracy.
- The camera stays on while you are in the Eye lab (red pulsing dot in the bar) and turns off when you press **Turn camera off** or leave the Eye lab.

## How accuracy is improved (and its limit)

1. **Continuous training**: ~16 samples per dot while your eyes rest on it, instead of a few clicks.
2. **13 calibration dots** covering the screen, in a snaking order.
3. **Second-stage correction**: the 5 check points (different from the training dots) are used to fit a small affine correction on top of WebGazer, applied only if a leave-one-out test shows it really helps (>= 10 % and >= 12 px better).
4. **Adaptive smoothing** (One Euro filter) for the blob: calm when your eyes rest, quick when they jump.
5. **Honest numbers**: the tracker bar shows measured error in pixels, not a vague label.

Webcam eye tracking has a physical floor. WebGazer-class trackers are typically 3-4 degrees off (about 100-200 px on a laptop), and even a good setup rarely gets below ~80 px. Judge by *region* (face? headline?), not exact pixels. This is why headline accuracy numbers come from public eye-tracking datasets (M12), and the webcam lab is the live demo and a pilot sanity check.

## Getting the best out of a webcam

- Face a light source; no bright window behind you. Sit 50-70 cm away, face inside the small camera box.
- Keep your head still and move your eyes. Use the same posture for calibration and viewing.
- Use a bigger, steadier window: calibration is stored per screen size, and runs use fullscreen when the browser allows it.
- If the check says "poor", recalibrate; do not keep going with a bad one.

## Running the pilot properly (PRD section 8)

- **The person who picked the thumbnails must not take the test.** Testers see each image for the first time.
- Every tester sees the **same set**, once each, in random order (the lab randomizes for you).
- 60-100 thumbnails per person for the click test, 15-20 for the webcam test (3 s each).
- Give each tester an anonymous label in the *Participant* box so sessions can be pooled.
- Present results as a **pilot / sanity check**, not the headline accuracy number.

## Reading the numbers

- **CC**: correlation of the predicted map with the real viewer map. -1 to 1, higher is better. One viewer is noisy; pooled viewers are steadier.
- **NSS**: how strongly the prediction lights up where people looked. 0 = chance, above 1 = solid.
- **First look**: did the real first fixation land within 160 px of our predicted fixation #1 (or one of our first three)?

Saved sessions can be pooled per thumbnail: `GET /gaze-sessions/<thumbnail id>`.

## Privacy

Webcam video is processed in the browser and never uploaded. Only gaze coordinates (x, y, time) go to your local server. The saved calibration (small eye-patch images plus screen positions, no full frames) lives in this browser's localStorage; **Forget it** deletes it.

## Offline use

WebGazer and its face-mesh model are copied into `frontend/public/` by `npm run sync:webgazer` (runs automatically before `dev` and `build`), so nothing is fetched from the internet.
WebGazer is **GPL-3.0** (LGPL-3.0 for companies valued under $1M). Fine for a demo; check the terms before distributing the app or the Chrome extension commercially.
