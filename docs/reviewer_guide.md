# Reviewer guide

For people who check results rather than run the pipeline. No programming needed.

## What the pipeline does

For each knee MRI, the pipeline finds the cartilage on the end of the thigh bone (*femoral cartilage*, FC) and on
the top of the shin bone (*tibial cartilage*, TC), measures how thick it is everywhere, and draws that thickness as
a flat map, the same shape for every knee, so knees and visits can be compared.

## The status table

An engineer runs `oai-analysis status` on a project, or sends you the table as a spreadsheet (`--csv`). It has one
row per knee image. The case name says whose knee and when: `9016304_96M_LEFT` is patient 9016304, the 96-month
visit, left knee.

| Status | Meaning | What to do |
|---|---|---|
| **Succeeded** | Finished, and every automatic check passed. | Nothing. Spot-check a few. |
| **Needs Review** | Finished, but an automatic check found something unusual. | Look at the case (below). |
| **Failed** | A step could not be done. The results are missing. | Tell an engineer if it keeps failing after a rerun. |
| **Outdated** | A setting changed since the case ran; the results are from the old setting. | Wait for the rerun. |
| **Incomplete**, **Not started**, **Running** | Not finished yet. | Wait. |

The columns `prep` to `bio` are the steps, in order: preparing the image, finding the cartilage (`seg`), lining the
knee up with the reference knee (`reg`), building the cartilage surface (`mesh`), measuring the thickness (`thick`),
carrying it onto the reference knee (`atlas`), and drawing the maps (`bio`). `✓` is done, `!` is where a check
flagged something, `✗` is where it failed.

## What *Needs Review* means

The automatic checks look for signs that a result may be wrong. They do not decide that it *is* wrong: a flagged
case can be fine, and that is what a reviewer decides. The note says what was found, for example:

| Note | Usually means |
|---|---|
| *37.6% of the femoral cartilage is outside its largest piece* | The cartilage was found in several separate pieces. Often a piece of something else was taken for cartilage, or the cartilage has a real gap. |
| *the tibial cartilage has no thickness on 18.0% of the atlas surface* | Part of the map is empty or zero: some of the cartilage was missed. |
| *the network's uncertainty about the tibial cartilage is 0.30* | The program was unsure where the cartilage is, e.g. a blurry or unusual image. |
| *after registration, the image correlates with the atlas at 0.720* | The knee was not lined up well with the reference knee, so the map may put thickness in the wrong place. |
| *the femoral cartilage volume is 2.1 ml (usual: 4.0-20.0 ml)* | Much less (or more) cartilage was found than in a typical knee. |
| *the tibial cartilage is thicker than 6.0 mm on 12.0% of its surface* | Implausibly thick cartilage, usually from a segmentation error. |

When there are several findings, the table shows the first one and "(and N more)"; the spreadsheet has all of them.

## Looking at a case

Each case has a `thumbnails` folder with three small pictures, in `cases/PATIENT/VISIT_KNEE/thumbnails/` (an
engineer can send them):

- `segment_overlay.png`: a slice of the MRI with the cartilage that was found, femoral in red and tibial in blue.
  Check that the colours sit on the cartilage, cover all of it, and nothing else.
- `atlas_map_FC.png` and `atlas_map_TC.png`: the thickness maps, all on the same 0–5 mm colour scale. Check for
  large empty or very dark patches, sharp jumps, or a pattern unlike the other knees.

Comparing with the same patient's other visit or other knee is often the quickest way to see whether something is
off.

## Reporting a case

Send the engineers:

1. the **case name**, e.g. `9016304_96M_LEFT`;
2. the **step** that was flagged or failed (the `!` or `✗` column, or the note);
3. the **note**, copied from the table;
4. a **screenshot** of what looks wrong (the thumbnail, marked up if you like) and a sentence on what you see;
5. your verdict: **looks fine**, **wrong**, or **not sure**.

"Looks fine" is useful too: it helps tune the checks so they flag fewer good cases.
