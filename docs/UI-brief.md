Files (ui/)
File	Size	What's in it
index.html	~160 lines	The static structure: header bar (Eluate mark and wordmark, the app-level controls), one graph card (the hero card: the track row, then Track level / Spectrum tabs), Controls (collapsible), transport footer. The mark is inline SVG.
style.css	~430 lines	All styling. Design tokens are at the top in :root (colours, type scale, gap). Archivo is loaded locally with @font-face.
app.js	~1200 lines	Builds the two charts as SVG, fills in values, wires the controls, and talks to the engine when running inside the app.
fonts/Archivo.ttf + OFL.txt	 	Variable font, width 62–125 and weight 100–900; its OFL licence ships with it.
data.js	(ignored, local)	One real render's numbers, for previewing in a browser.
Working on it standalone
1. Generate preview data from any finished render:
.venv/Scripts/python -m eluate.uidata "work/<track> - eluate" ui/data.js
2. Open ui/index.html straight in Chrome or Edge. Before/After playback works too, since it reads that render's ab/ files.
3. Add #controls to the URL (index.html#controls) to open with the Controls panel expanded.
The browser shows the finished-render state only. The empty state ("No track"), the preview while rendering, the stage/elapsed text and the error line only appear inside the app. To see those, run cd desktop, then npm run dev (it uses the .venv engine directly). There's no hot reload: each start copies ui/ into the app, so restart it after edits.
What you can change freely
- Colours, type and spacing: everything in :root. The SVG charts use the same CSS variables (fill: var(--ice) and so on), so palette changes carry through to the charts.
- Layout, spacing, borders, button styles, the disclosure chevron, footer, header: all plain CSS.
- Chart text: styled through the classes .tl-axis, .tl-label, .sec-name, .sec-ref, .sec-val. The legends are HTML (.spec-legend, .key) in each view's head row, not SVG.
What app.js depends on (keep or tell me)
- Element IDs: open, track, delivery, stage, stage-text, stage-time, cancel, log, reveal, cpu-wrap, cpu, cpu-l, render, tl, readout, disclosure, summary, controls-h, controls-body, tilt, tilt-v, tilt-got, preserve, preserve-l, preserve-v, bpm, bpm-v, bpm-got, character, loud, loud-v, loud-got, spectrum, view, spec-tilt, sp, play, ab, pos, stats, a-before, a-after, graph, sp-readout.
- Segmented buttons: .seg buttons with role="radio", aria-checked and data-v (delivery, profile, view, A/B, character, graph).
- Graph card: JS sets data-view="level" or "spectrum" on .stage-card (the graph tabs); CSS shows one view and hides the other with visibility, so both stay laid out and draw at full width. Keep it visibility, not display or hidden, or the hidden chart draws at width 0.
- Readouts: #readout (timeline hover) and #sp-readout (crossover / side high-pass hover) sit fixed at the right of their view's legend row. JS only fills them; it no longer positions or hides #readout, which keeps the last position, dimmed, after the pointer leaves.
- Crossover handles: .xo, plus .side on the side high-pass; their line and label colours come from CSS. JS sets aria-checked, and the CSS styles off it.
- Classes JS toggles: changed on Render; running and error on the stage text; none on the track name while no track is open; locked on crossover handles during a render; short (amber) on reached values, the summary and stats; drop on .timeline while a file is dragged over it; and got, of, item, k, v, which JS writes into the reached values and the Controls heading summary.
- The hidden attribute shows and hides the Log, Open folder, Limit CPU (app only), spectrum section and controls body ([hidden] { display: none !important }).
- Chart geometry lives in JS, not CSS: the TL and SP constants near the top of the timeline and spectrum sections set strip heights and margins. .tl { height: 318px } must stay in step with TL (the time axis sits at y 312). The spectrum height comes from .sp. If you want different proportions, change both, or tell me and I'll make the charts follow the CSS.
Constraints inside the desktop app
- CSP: no inline <script> or onclick= attributes, and nothing loaded from the web: no Google Fonts or CDNs. Fonts and images must sit in ui/, and images can also be inline data: URIs. Inline style="" attributes are allowed.
- The brief's GUI text rule: control names, live values, must-know state and navigation only. No taglines, hints or explanatory copy.
- Window size: opens at 1440×980, minimum 960×640. There's also a layout below 900 px wide, mostly irrelevant in the app.
After your pass
Hand me the files, or just edit ui/ in place. I'll check nothing JS relies on has broken, then rebuild the installer (npm run build only; the engine doesn't need rebuilding for UI changes).

Built 2026-10-21 (sprint WP1 to WP3): yours to restyle, keep the ids and roles
- The track list (`#list`, the listbox `#rows`, rows `.trk` with `aria-selected`, state classes `st-queued` / `st-running` (a render, or the track's analysis) / `st-done` / `st-error`, cells `.nm`, `.st`, `.num`, `.num.short`, the remove button `.rm`), shown only when two or more tracks are open. Columns: name, state, loudness reached, tilt, crest (before small), true peak, flag count.
- The Folder button (`#folder`) beside Open; Open takes several files. Progress "track 3 of 10" after the stage text while a list renders.
- Tempo, crossovers and the side high-pass are per track: the controls edit the selected row.
- Amber comes from the engine's flags (`D.flags`); app.js only decides where a flag shows.
- The limiter-GR fill is ice (R5); the screen-reader descriptions are one clause each (R6).

- Limit CPU beside Render (sprint WP8): `.cpu` (`#cpu-wrap`) holds the name `#cpu-l` and a `.toggle` switch `#cpu` (role switch, aria-checked), the same switch as Preserve mix. Off: all cores but two on one render (tracks render one at a time; `PARALLEL_RENDERS` in app.js would allow up to three at once).

Progress, lock-out and Cancel (maintainer feedback 2026-10-24: "it starts analysing immediately ... no estimate ... no lock out ... or cancel button")
- The status in the track row (`#stage`, under the name) says what the engine does in plain words: "Reading track" while a track loads (the preview: the file read and the before graphs, no analysis), then "Analysing" while its analysis runs in the background (the analysis the render needs, cached so Render does not repeat it), then "Ready"; during a render "Step 1 of 4: Analysing", "Step 2 of 4: Planning", "Step 3 of 4: Mastering · render 2 of up to 7", "Step 4 of 4: Writing report", "Loading graphs", then "Done". `#stage-text` is the live region (role status; alert for an error), announced when the stage changes; `#stage-time` beside it is the engine's estimate counted down ("about 6 min left", "under a minute left"), or the elapsed time when there is none, and is not announced. `.stage.running` keeps the spinner and lights the text.
- Cancel (`#cancel`, a `.ghost` button at the right of the track row) shows while a render runs or the selected track is being analysed; it stops the render (or the queue) and the analysis. The stopped track goes back to Queued (a Done track too once its re-render had started: the engine clears the completion marker first). Render stays Render and is disabled while a render runs; focus moves to Cancel when Render is pressed, back to Render after.
- While a render runs (or the analysis it waits for), the settings it renders with are disabled with a visible dimmed state: Tilt, Preserve mix, Tempo, Character, Loudness, Delivery, Profile and the crossover / side high-pass handles. Playback, the graph tabs, the list (selecting rows), Open / Folder / drop (new tracks join the queue; a file already in the list is ignored) and Limit CPU stay usable.

Header and track row (maintainer feedback 2026-10-24: the name and the status were cut in the header at the default window size; "should those live in the hero area?" yes; "single track should wait")
- The header holds the app-level controls only: brand, Open (plain "Open", several files; `.open`, the look the removed empty-state "Open track" button had: well background, brighter border and text, reach border on hover, so it reads as the way in; the empty graph shows no button), Folder, Delivery, Profile, then Limit CPU and Render at the right. Below 1180 px the brand takes a row of its own and the controls the row under it.
- The hero card (`.stage-card`) starts with the track row (`.track-row`, above the graph tabs): the selected track's full name (`#track`, an h1 at title weight; "No track", muted, before one is open; ellipsis only when the window cannot fit it, the full name in its title), the status line under it (`#stage`: `#stage-text`, the live region, and `#stage-time`), and the track's actions at the right (`.track-actions`: Cancel, Log, Open folder). The actions wrap under the name before the name is cut. At 1440 px the longest names and statuses (e.g. "Step 3 of 4: Mastering · render 2 of up to 7 · track 1 of 3 · about 12 min left") fit on one line each.
- The row is the selected track's: its own render, analysis or loading, else its state (Ready, Queued, Done, Error). While the queue renders another track the status adds which ("Queued · rendering track 1 of 3"); Cancel still stops the queue.
- Nothing renders by itself: one track opened alone loads ("Reading track"), analyses in the background ("Analysing"), then waits at "Ready" for Render, as a list does.

Built in the 2026-10-21 sprint (WP5), was listed here as coming
- The Electronic / Live segmented control (`#profile`, a `.seg`) beside Delivery in the header (the live profile).
