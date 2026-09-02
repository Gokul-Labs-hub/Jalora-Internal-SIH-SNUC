# Groundwater Depletion & Salinity Infiltration Predictive Matrix

A predictive analytics dashboard that correlates simulated DWLR (Digital
Water Level Recorder) readings, rainfall, and agricultural pumping data to
forecast aquifer drawdown and coastal salinity ingress — with **two
interfaces in one app** (Water Authority / Resident-Public) and an
**Emergency Alert & Broadcast module**.

## Latest update: crop suggestions for coastal farmers (Public Interface)

A new "🌾 Crop Suggestions for Your Area" section appears on the Public
Interface — **coastal wells only** (matching the request scope), driven by
that well's current risk band:
- **High risk**: salt-tolerant crops — cotton, sapota/guava, coconut, plus
  a pointer to ask about ICAR-CSSRI's salt-tolerant seed varieties of
  paddy/wheat/chickpea/mustard.
- **Medium risk**: low-water + moderately salt-tolerant crops — millets
  (bajra/ragi/jowar), groundnut, chilli, coriander/fennel.
- **Low risk**: proactive low-water crops (millets, pulses) to help keep
  the area healthy.

**Grounded in real research, not invented**: every crop listed is one
ICAR-CSSRI (India's national salt-affected-soil research institute)
actually identifies for salt-affected coastal land, based on their
coastal regional research station findings (Canning Town RRS) and
crop-improvement work. I verified this via a live search before writing
any of it, rather than relying on general assumption.

**Honesty note, stated in-app too**: CSSRI's published tolerance figures
are for *soil* salinity (ECe), while this app's EC reading is a
*groundwater* salinity proxy — related but not the same measurement, and
actual soil salt buildup also depends on soil type and drainage. The
in-app caption says so and points farmers to their local agriculture
office to confirm for their exact field, rather than presenting this as a
precise field-specific prescription.

**Scope note on translation**: crop names (Bajra, Ragi, Cotton, etc.) are
kept in their widely-used pan-Indian form rather than translated into
each script — regional botanical terminology varies by dialect in ways
this app can't verify, so one consistent, nationally-recognized form is
more reliable than a possibly-inaccurate per-language translation. The
surrounding guidance text and citation *are* fully translated, verified
clean with the same script-contamination check used earlier.

## Latest update: multi-language support (Auto + Manual)

**Public/User Interface**: automatic language switching based on the
selected coastal zone's state language (Tamil Nadu → Tamil, Kerala →
Malayalam, West Bengal → Bengali, Odisha → Odia, Gujarat → Gujarati,
Andhra Pradesh → Telugu, Karnataka → Kannada), plus a manual override
dropdown — pick a specific language and it stays chosen regardless of
which area you view next, until you switch back to "Auto" or pick
another. Inland wells default to English in Auto mode (the auto-mapping
is scoped to "coast," per how this was requested). Every string a
resident sees is translated — title, alerts, status messages, advice.

**Authority Interface**: manual-only switching (no auto-switch — officials
aren't forced into a language just by clicking into a particular well),
defaults to English. The main chrome (titles, section headers, the
SEND BROADCAST button, core alert/status text) is translated. Deep
technical content — formulas, unit citations, methodology, feature-
importance labels — stays in English by design: this matches how real
regional-language government tools work (scientific terminology is
conventionally kept in English even in a localized UI) and avoids
machine-translating a formula or citation in a way that could distort it.

**Honesty notes:**
- These are AI-drafted translations for a prototype demo, not
  professionally localized or native-speaker-verified. Recommend review
  by a native speaker of each language before any real deployment,
  especially Odia (the language I have the least training exposure to).
- **A real bug was found and fixed during my own verification**: my first
  draft of the Odia language-picker label was accidentally corrupted with
  three Bengali-script characters mixed into otherwise-correct Odia (an
  easy mistake given how visually similar the two scripts can look while
  actually using different Unicode code points). I caught this by
  programmatically checking every character's Unicode block against the
  expected script for all 8 languages — not by eyeballing — which is why
  I'm confident saying the rest is clean: all 432 translated strings
  across the 8 non-English languages were verified, character-by-
  character, to fall in their correct Unicode script block, with the one
  found error corrected. This kind of verification is exactly what a
  judge would want to hear you can explain if asked "how do you know
  your translations aren't broken."

## Latest update: data-credibility & validation audit
This pass corrected several scientific-honesty gaps rather than adding
features: real chronological train/validation/test metrics (R²/MAE/RMSE,
computed, not invented) now saved and shown in-app; extraction figures are
rounded to avoid false precision and carry a computed confidence
indicator; a data-provenance caption (source/period/resolution/unit/
Observed-Derived-Estimated) is attached to every major metric; High/
Critical risk now shows real, data-computed contributing factors; and a
real date-consistency bug (broadcast timestamps using wall-clock time
instead of the dataset's own date range) was found and fixed. Full details
in the audit report delivered alongside this update — ask if you don't
have it and want the summary repeated here.

## Run it (do these in order, every time you set up on a new machine)

```
python data_generator.py
python train_model.py
streamlit run app.py
```

Nothing in this update touched `data_generator.py` or the model-training
logic in `train_model.py` — only `app.py` changed. If you already have
`groundwater_data.csv` and the three `.pkl` files from before, you can
skip straight to `streamlit run app.py`.

## What's new in this update

### 1. Theme & animation polish
A gradient hero banner on the landing screen (floating water-drop
animation), styled choice cards, hover-lift buttons, and softly-shadowed
metric cards site-wide — cosmetic only, no logic changed.

### 2. Refined electricity-based extraction estimate — no more arbitrary constants
The previous version used a flat η = 0.40 and a flat +5m head allowance.
Both were arbitrary and you were right to push back. Here's what changed,
and what's genuinely achievable vs. not:

**What I could not do, and why:** you asked for the actual aggregate
agricultural electricity-consumption data from TANGEDCO at the correct
spatial/temporal resolution. I checked this before building anything (not
just once, but again for this update): **Tamil Nadu supplies agricultural
electricity free of charge, most agricultural connections are unmetered in
practice, and TANGEDCO does not publish an open feeder/block/district-level
agricultural-consumption dataset.** There is no real dataset at any
resolution to fetch here — this isn't a limitation of my access, it's a
limitation of what exists publicly. Claiming to use "actual TANGEDCO data"
would mean fabricating numbers, which you've correctly told me not to do
twice now, and I'm holding to that.

**What I did instead — a properly grounded, defensible estimate:**

- **η (pump efficiency)** is no longer a single guessed number. You choose
  a pump-type profile, and the app uses the low/mid/high efficiency range
  from **published Indian field studies**, not an assumption:
  - Conventional/older pumpsets (most common in the field): **20–35%**
    overall (wire-to-water) efficiency. Sources: World Bank Haryana
    pump-energy audit (21–24%); National Productivity Council Haryana
    study (25–35%); a 65-tubewell field survey in Sonipat measuring
    10.1–56.6% ([PMC9884473](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9884473/));
    [ScienceDirect review of Indian agricultural pumping efficiency](https://www.sciencedirect.com/science/article/abs/pii/S0973082608601787)
    (~20–30%, citing Dixit & Sant 1996 and Singh 2009).
  - BEE 5-star-rated modern pumpsets: **40–50%**.
- **H (Total Dynamic Head)** is no longer a flat +5m guess. Static lift
  uses this well's own **real measured DWLR depth** (already in the app,
  not new), and the delivery/friction component is now expressed as
  **10–25% of that static lift** — a standard hydraulic-engineering
  approximation that scales with well depth, instead of adding the same
  fixed 5 metres to a 5m-deep well and a 25m-deep well alike.
- **Uncertainty is now properly propagated**, not just offset by ±0.10:
  the low estimate pairs the least-favourable combination (low η, high
  head), the high estimate pairs the most-favourable (high η, low head),
  and the "Estimated Agricultural Groundwater Extraction" figure is
  reported as **low – mid – high**, with units, every time.
- **Attribution caveat, made explicit and adjustable:** official
  "agricultural electricity" categories (where metered at all) can include
  non-pumping farm loads. There's now a "% of entered electricity
  attributable to groundwater pumping" input (default 100%, lower it if
  your figure is broader than pumping alone) — the app no longer silently
  assumes the whole figure is pumping.
- **Physical plausibility cross-checks**, shown alongside the estimate:
  compared against this well's own historical 30-day pumping range,
  against recent rainfall (flags if heavy pumping is claimed alongside
  heavy recent rainfall — less typical), and against the well's 90-day
  depth trend.
- Find all of this in the Authority view's **"⚡ Estimated Agricultural
  Groundwater Extraction"** panel and the electricity sidebar section.

### 3. Map refinement
- A prominent, animated red banner now appears above the map whenever any
  well is projected at High risk, naming the zones needing priority
  attention.
- High-risk wells are now drawn noticeably larger on the map than their
  drawdown magnitude alone would size them (a ~2.8x boost vs. Low), so red
  = danger is visually unmissable rather than just another dot colour.

### 4. Emergency Alert & Broadcast module (Authority Interface)
A new **"📢 Emergency Alert & Broadcast"** section, prominently placed
right under the top status metrics:

- **Automatic Critical Alert**: for whichever well/severity you're
  viewing, a plain-language message is auto-drafted with **zero ML/
  technical jargon** — e.g. *"CRITICAL GROUNDWATER ALERT — High
  salinity-ingress risk has been detected in the Chennai Coast coastal
  monitoring zone. Residents and agricultural users in the affected area
  should reduce groundwater extraction and follow the latest authority
  guidance."* It's editable before sending.
- **Authority's Manual Message**: a separate free-text box for local
  context or specific instructions (e.g. "water tankers arriving
  tomorrow").
- **Severity wording follows the safety rule you specified**: the system
  never claims a confirmed disaster — it says *"detected by the monitoring
  system"* / *"predicted risk"*, consistently.
- **Recipient targeting**: the zone matching your currently-selected well
  is auto-suggested by default ("All in {zone} zone"), with options to
  broadcast to the entire region or hand-pick individuals instead. You
  always have final control before anything sends.
- **SEND BROADCAST** button: on click, the alert is (1) generated, (2)
  stored in Broadcast History, (3) made immediately visible in the
  Public/User Interface, and (4) an honest delivery status is shown —
  see the email section below for exactly what "honest" means here.
- **Manage Recipients**: add/edit/remove people (Name, Zone, Email, Phone
  optional, Category — Resident/Farmer/Local Official). Seeded with a
  couple of example contacts per coastal zone so the demo isn't empty on
  first run; edit freely.
- **Broadcast History**: every past broadcast — date/time, severity,
  location, message, recipient count, delivery status.

### 5. Email — real per-recipient delivery, configured honestly, never faked
This update sends the final broadcast **individually to every selected
recipient's own email address** (not one email to a group) and tracks
**each recipient's result separately as Sent or Failed** — shown in the
Authority Interface right after you click Send, e.g. *"Broadcast
delivered: 2/2 emails successfully sent."* Every email contains the
affected location, severity, the automatic warning, your manual message
(if any), a recommended action, and the timestamp.

**No credentials are hard-coded anywhere** — the app reads five
environment variables and nothing else. If they're not set — the default,
expected state until you configure them — every broadcast shows exactly:
**"External email service not configured."** It never claims an email was
sent when it wasn't, and if some recipients succeed while others fail
(e.g. a bad address), that's reported honestly as a partial result, not
rounded up to "sent."

**Important, and tested directly:** I do not have outbound internet access
in the environment I build in, and I don't have a real Gmail account to
send through — so what I've verified is the *logic*: using simulated SMTP
responses, I confirmed correct behaviour for "not configured," full
success, full failure (bad password), a mix of success/failure across
recipients, and both port 465 (SSL) and 587 (STARTTLS) connection modes.
**Genuine end-to-end delivery through a real mail server can only be
confirmed by you**, using your own credentials, following the steps below.

#### How to configure real Gmail delivery (5–10 minutes)

1. **Use a dedicated Gmail account for this**, not your personal one (a
   free new Gmail account is fine — you're about to generate a password
   specifically for this app).
2. **Turn on 2-Step Verification** on that account: 
   [myaccount.google.com/security](https://myaccount.google.com/security) → "2-Step Verification" → follow the prompts.
   (Google requires this before it will let you create an App Password.)
3. **Create an App Password**:
   [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
   → give it any name (e.g. "groundwater-app") → click Create. Google
   shows you a **16-character password** — copy it now, you won't see it
   again (you can always generate a new one later if you lose it).
4. **Set the five environment variables** in the same Terminal window,
   **before** running `streamlit run app.py` (Mac/Linux):
   ```
   export SMTP_HOST=smtp.gmail.com
   export SMTP_PORT=587
   export SMTP_USERNAME=your.dedicated.account@gmail.com
   export SMTP_PASSWORD=the16characterapppassword
   export SENDER_EMAIL=your.dedicated.account@gmail.com
   streamlit run app.py
   ```
   (`SMTP_USERNAME` and `SENDER_EMAIL` are normally the same Gmail
   address; `SMTP_PASSWORD` is the App Password from step 3, **not** your
   normal Gmail login password — your regular password will not work
   here.) These `export` lines only last for that terminal session — you'll
   need to re-run them (or add them to your shell profile) next time you
   open a new terminal.
5. **Test it**: open the Authority view, pick a coastal well with at least
   one recipient that has a real email address you can check (edit one of
   the seeded recipients under "Manage Recipients" to use your own email
   for this test), click **SEND BROADCAST**, and check that inbox.
6. If it doesn't arrive: check the "Per-recipient delivery detail" table
   that appears right after sending — it shows the *exact* error Gmail's
   SMTP server returned (wrong password, blocked login, etc.), which is
   usually self-explanatory. Common ones: **"Username and Password not
   accepted"** almost always means you pasted your normal password instead
   of the App Password, or 2-Step Verification isn't actually turned on
   yet.

### 6. Active Authority Alerts (Public/User Interface)
A new section shows real broadcasts (not simulated) for whichever location
the resident has selected — severity, location, date/time, the automatic
warning, the authority's additional message, and delivery status, using
the same RED/ORANGE/YELLOW/GREEN legend. The page automatically checks for
new alerts every 20 seconds (a lightweight page refresh — no separate
server needed), plus a manual "🔄 Check now" button for an instant check
during a live demo.

**Why this works across the Authority and Public tabs simultaneously:**
alerts and recipients are held in a shared, in-memory store
(`st.cache_resource`) that every browser tab connected to the same running
`streamlit run app.py` process sees — so a broadcast sent from the
Authority tab appears in the Public tab without needing a database. This
resets if you restart the Streamlit server, which is a normal, documented
limitation for a prototype (production would use a real database).

## How the prediction actually works (unchanged from before)

1. **Depth model**: Random Forest, predicts water-table depth 30 days
   ahead. R² ≈ 0.996, MAE ≈ 0.31m on a held-out 7-month test window.
2. **Salinity model**: Random Forest, predicts EC 30 days ahead from the
   projected depth, chained from model 1. R² ≈ 0.98, MAE ≈ 36 µS/cm, plus
   a small, capped, coastal-only, documented adjustment for sustained
   over-pumping (tree models can't extrapolate strongly past their
   training range — see the in-app "Formulas used" panel for the exact
   equation).
3. **Risk bands** (unchanged core model): Low <1200, Medium 1200–1600,
   High >1600 µS/cm. The Broadcast module's four-tier RED/ORANGE/YELLOW/
   GREEN legend is a display-only extra split built from this same EC
   value and the same calibrated cutoffs — it does not replace or
   redefine the core three-tier risk model used elsewhere in the app.

## Talking points for your demo / PPT

- Landing screen: "one tool, two audiences."
- Authority view: show the current status, open the extraction-estimate
  panel and mention the TANGEDCO finding (it shows real domain research,
  not just modelling), then scroll to the Broadcast module — pick a
  high-risk well, show the auto-drafted plain-language message, add a
  manual instruction, hit Send.
- Switch to Public view (or a second browser tab) and show the alert
  appear there in real time — this is a strong, concrete "wow" moment.
- Point at the map's red danger banner and oversized High-risk markers.
- Be upfront about the synthetic base data, the stress adjustment, and the
  TANGEDCO data-availability finding — all explained above and in-app.
  Judges respect this far more than an unexplained "verified real-time
  data" claim.

## How to deploy (get a public link judges can open on any device)

**What changed to make this possible:** the app now auto-generates
`groundwater_data.csv` and the three `.pkl` model files on first run if
they're missing (takes ~20 seconds, only happens once) — verified with a
full cold-start test. You do **not** need to run anything locally first or
commit the generated data/model files (they're ~70MB combined, awkward for
a git repo) — just push the 5 source files below.

### Step 1 — Put the code on GitHub
1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Create a new repository (public or private both work).
3. Upload exactly these 5 files: `app.py`, `data_generator.py`,
   `train_model.py`, `requirements.txt`, `README.md` — via "Add file →
   Upload files" in the browser, no command line needed. Do **not** upload
   any `.csv` or `.pkl` files — the app creates them itself.

### Step 2 — Deploy on Streamlit Community Cloud (free)
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
2. Click **"New app"**, pick your repository, branch (`main`), and set the
   main file path to `app.py`.
3. Click **"Deploy"**. The first load will show a "Setting up for the
   first time..." spinner for about 20 seconds while it generates the data
   and trains the models — this is normal and only happens once per
   deployment (or after the app has been asleep long enough that Streamlit
   Cloud recycles its storage — see Troubleshooting below).
4. You'll get a public link like `https://your-app-name.streamlit.app` —
   this works on any device, no laptop required at demo time.

### Step 3 — (Optional but recommended) enable real email broadcasts
Without this, the broadcast feature still works fully in-app — it just
shows "External email service not configured," which is honest and fine
for a demo. To enable real email:
1. Get a Gmail App Password — see the "How to configure real Gmail
   delivery" section above (steps 1-3) if you haven't already.
2. On your app's page at share.streamlit.io, click the **⋮** menu →
   **Settings → Secrets**.
3. Paste in this exact format (TOML — no quotes needed around the
   values shown here, though quoting also works):
   ```
   SMTP_HOST = "smtp.gmail.com"
   SMTP_PORT = "587"
   SMTP_USERNAME = "your.dedicated.account@gmail.com"
   SMTP_PASSWORD = "the16characterapppassword"
   SENDER_EMAIL = "your.dedicated.account@gmail.com"
   ```
4. Click **Save** — the app restarts automatically with these available.
   The code checks `st.secrets` first, then falls back to plain
   environment variables, so this works whether you deploy on Streamlit
   Cloud or elsewhere. **Never commit these values into the repo itself**
   — Secrets is specifically for values that must stay out of your code.

### Alternative hosts
Streamlit Community Cloud is purpose-built for this and the easiest path,
but the app also runs on any host that can run a Python web process —
[Hugging Face Spaces](https://huggingface.co/spaces) (free, supports
Streamlit directly) and [Render](https://render.com) are both viable if
you want a second option. On those, set the same 5 `SMTP_*`/`SENDER_EMAIL`
values as plain environment variables in that platform's dashboard rather
than Streamlit's Secrets — the fallback in the code handles that
automatically.

### Troubleshooting deployment
- **App shows the setup spinner every time you open it, not just once**:
  Streamlit Community Cloud's free tier can put an inactive app to sleep
  and clear its local disk; waking it re-triggers the ~20-second setup.
  This is a normal free-tier trade-off, not a bug — the app still works
  correctly, it just re-does the one-time setup after long idle periods.
- **Setup spinner fails with an error**: click "Manage app" → "Logs" on
  Streamlit Cloud to see the real Python error — usually a
  `requirements.txt` install issue. Confirm all 5 files were uploaded and
  `requirements.txt` wasn't accidentally edited.
- **Broadcast emails don't send after adding Secrets**: double-check you
  used `SMTP_USERNAME` and `SENDER_EMAIL` (not `SMTP_USER`/`SMTP_FROM` —
  older docs/versions of this project used those names, the app now reads
  the names shown in Step 3 above) and that the password is the 16-character
  App Password, not your normal Gmail password.

## Troubleshooting

- **"python is not recognized"** (Windows): reinstall Python from
  python.org, tick "Add python.exe to PATH."
- **Streamlit shows the red "Missing file(s)" message**: run
  `python data_generator.py` then `python train_model.py` first.
- **Port already in use**: `streamlit run app.py --server.port 8502`.
- **A broadcast sent in one tab doesn't appear in another**: make sure
  both tabs are pointed at the same running `streamlit run app.py`
  process (same terminal/port) — restarting the server clears the
  in-memory alert/recipient store by design (see note above).
- **Email always shows "not configured"**: expected unless you've set the
  five `SMTP_*` environment variables described above before launching.
