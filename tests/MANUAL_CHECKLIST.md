# Manual Hermes regression checklist

Run after refactors on a known Hermes 1D case and a known Hermes 2D case.

## 1D case

- [ ] `python holo-shed.py /path/to/1d/case` launches without errors
- [ ] Case loads; variable list populated
- [ ] Select one or more variables; profile plot updates
- [ ] Time slider / spinbox changes time slice
- [ ] Y-scale and y-limit mode toggles work (context menu + overlay buttons)
- [ ] Add second case (overlay); colours / linestyles distinct
- [ ] Time history tab redraws for selected variables

## 2D case

- [ ] `python holo-shed.py /path/to/2d/case` launches in 2D mode
- [ ] Poloidal extract tab: region, sepadd, time slider work
- [ ] Radial extract tab: region and overlays work
- [ ] 2D field (polygon) tab: variable, cmap, log scale, time slider
- [ ] Monitor tab: OMP / target traces vs time
- [ ] “Show cut in 2D” overlay opens and tracks cut location
- [ ] Multi-case overlay on 2D extract tabs

## Backend / SOLPS stub

- [ ] Directory with only `balance.nc` shows clear NotImplementedError (no silent wrong plots)
