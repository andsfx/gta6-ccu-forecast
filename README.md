# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/andsfx/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/andsfx/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A deterministic Python 3.13 forecaster for GTA VI's launch-day Twitch viewer counts. It reads the case file (everything recorded up to the end of release day's first hour) and writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24, scored by pinball loss on the log scale.

The model pools release-day curves of comparable console-at-launch launches, normalised to each launch's own hour-1 print, and combines Gaussian quantiles with weighted empirical quantiles. A momentum correction shifts hours 5 to 24 up when hour 1 printed below the pre-launch hour 0 (the EU unlock wave had not landed yet).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit input dir> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with standard library only. Two runs on the same case write byte-identical forecasts.

---

<sub>Entered on [make.fun](https://make.fun/?ref=r76jbz76&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
