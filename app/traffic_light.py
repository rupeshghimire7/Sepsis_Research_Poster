"""Traffic-light prototype for the poster stand. Research prototype, not for clinical use. Invented patients only.

python app/traffic_light.py                      # uses out/models/bundle.joblib, http://localhost:8050
python app/traffic_light.py --bundle path/to/bundle.joblib --port 8050

Runs offline with the standard library: enter triage details, get a low / moderate / high tier, the calibrated
risk, and the three strongest reasons (TreeSHAP contributions from the XGBoost model). Features are built by the
same code as training, so the app and the paper cannot drift apart.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sepsis_ed.cohort import clean_triage  # noqa: E402
from sepsis_ed.explain import reasons  # noqa: E402
from sepsis_ed.features import context_features, vital_features  # noqa: E402

WEEKDAY, SATURDAY = pd.Timestamp("2026-10-28"), pd.Timestamp("2026-10-31")  # any Wednesday / Saturday


def build_row(form, cfg, features):
    """One invented patient -> one-row feature frame, through the training feature code."""
    num = lambda k: pd.to_numeric(form.get(k), errors="coerce")  # noqa: E731
    day = SATURDAY if form.get("weekend") else WEEKDAY
    raw = pd.DataFrame([{
        "age": num("age"), "gender": form.get("sex", "F"),
        "temperature": num("temperature"), "heartrate": num("heartrate"), "resprate": num("resprate"),
        "o2sat": num("o2sat"), "sbp": num("sbp"), "dbp": num("dbp"), "pain": num("pain"), "acuity": np.nan,
        "chiefcomplaint": form.get("complaint", ""), "arrival_transport": form.get("arrival", "WALK IN"),
        "intime": day + pd.Timedelta(hours=int(num("hour") if pd.notna(num("hour")) else 12)),
    }])
    raw = clean_triage(raw, cfg)  # same unit fix and plausible ranges as training
    row = pd.concat([vital_features(raw, cfg), context_features(raw, cfg)], axis=1)
    row["ed_visits_365d"] = num("ed_visits") if pd.notna(num("ed_visits")) else 0
    row["admissions_365d"] = num("admissions") if pd.notna(num("admissions")) else 0
    row["prior_sepsis_admission"] = int(bool(form.get("prior_sepsis")))
    for f in features:
        if f.startswith("hx_"):
            row[f] = int(bool(form.get(f)))
    return row.reindex(columns=features), raw


class App:
    def __init__(self, bundle_path):
        self.bundle = joblib.load(bundle_path)
        self.fm = self.bundle["primary"]["xgboost"]
        self.cfg = self.bundle["config"]
        tl = Path(bundle_path).parents[1] / "table3_traffic_light.csv"
        self.tiers = pd.read_csv(tl).set_index("tier").to_dict("index") if tl.exists() else {}

    def predict(self, form):
        row, raw = build_row(form, self.cfg, self.fm.features)
        tier = str(self.fm.tier(row)[0])
        groups = [k.replace("cc_", "").replace("_", " ") for k in self.cfg["complaint_groups"] if row[k].iloc[0] == 1]
        cleaned = [v for v in ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")
                   if pd.notna(pd.to_numeric(form.get(v), errors="coerce")) and pd.isna(raw[v].iloc[0])]
        info = self.tiers.get(tier, {})
        return {"tier": tier, "risk": float(self.fm.risk(row)[0]), "reasons": reasons(self.fm, row, 3),
                "complaint_groups": groups or ["other"], "set_to_missing": cleaned,
                "tier_sepsis_rate": info.get("sepsis_rate"), "tier_share_of_arrivals": info.get("share_of_arrivals")}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sepsis Triage Traffic Light</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--banner:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--banner:#3a2f12}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;
--accent:#3987e5;--banner:#3a2f12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.banner{background:var(--banner);padding:10px 16px;text-align:center;font-weight:600}
main{max-width:1040px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:1.1fr 1fr;gap:20px}
@media (max-width:820px){main{grid-template-columns:1fr}}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}@media (max-width:480px){.grid{grid-template-columns:repeat(2,1fr)}}
label{display:flex;flex-direction:column;font-size:12px;color:var(--ink2);gap:4px}
input,select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:7px 8px;width:100%}
.wide{grid-column:1/-1}.checks{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink2)}
.checks label{flex-direction:row;align-items:center;gap:6px;font-size:13px}.checks input{width:auto}
fieldset{border:0;padding:0;margin:14px 0 0}legend{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
button{font:inherit;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:7px 12px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;margin-top:16px;width:100%}
.light{display:flex;gap:10px;align-items:center;margin:4px 0 14px}
.lamp{width:30px;height:30px;border-radius:50%;background:var(--line);opacity:.35}
.lamp.on{opacity:1}.lamp.low{background:var(--good)}.lamp.moderate{background:var(--warn)}.lamp.high{background:var(--crit)}
.tier{font-size:30px;font-weight:700;margin:0}.risk{color:var(--ink2);margin:2px 0 14px}
ol{padding-left:20px;margin:6px 0}li{margin:6px 0}.up{color:var(--crit)}.down{color:var(--good)}
.note{font-size:12px;color:var(--muted);margin-top:12px}.empty{color:var(--muted)}
</style></head><body>
<div class="banner" role="note">⚠ Research prototype, not for clinical use. Invented patients only.</div>
<main>
<section class="card">
<h1>Sepsis risk at triage</h1><p class="sub">Model trained on MIMIC-IV-ED (Boston, 2011–2019). Not validated in NSW.</p>
<div class="presets" aria-label="Invented example patients">
<button type="button" data-p="0">Example: febrile older adult</button>
<button type="button" data-p="1">Example: ankle injury</button>
<button type="button" data-p="2">Example: confused, low BP</button></div>
<form id="f">
<div class="grid">
<label>Age<input name="age" type="number" min="18" max="110" required></label>
<label>Sex<select name="sex"><option value="F">Female</option><option value="M">Male</option></select></label>
<label>Arrival<select name="arrival"><option value="WALK IN">Walk-in</option><option value="AMBULANCE">Ambulance</option>
<option value="HELICOPTER">Helicopter</option><option value="OTHER">Other</option></select></label>
<label>Temperature °C<input name="temperature" type="number" step="0.1"></label>
<label>Heart rate /min<input name="heartrate" type="number"></label>
<label>Resp. rate /min<input name="resprate" type="number"></label>
<label>SpO₂ %<input name="o2sat" type="number"></label>
<label>Systolic BP<input name="sbp" type="number"></label>
<label>Diastolic BP<input name="dbp" type="number"></label>
<label>Pain 0–10<input name="pain" type="number" min="0" max="10"></label>
<label>Hour of arrival<input name="hour" type="number" min="0" max="23" value="12"></label>
<div class="checks" style="align-self:end"><label><input type="checkbox" name="weekend">Weekend</label></div>
<label class="wide">Chief complaint (free text, as typed at triage)<input name="complaint" placeholder="e.g. fever, cough"></label>
</div>
<fieldset><legend>History before this visit</legend><div class="grid">
<label>ED visits, past year<input name="ed_visits" type="number" min="0" value="0"></label>
<label>Admissions, past year<input name="admissions" type="number" min="0" value="0"></label></div>
<div class="checks" id="hx" style="margin-top:10px"><label><input type="checkbox" name="prior_sepsis">Previous sepsis admission</label></div></fieldset>
<button class="primary" type="submit">Estimate risk</button>
</form></section>
<section class="card" aria-live="polite" id="out"><p class="empty">Enter triage details or pick an example.</p></section>
</main>
<script>
const HX = __HX__;
const hx = document.getElementById("hx");
HX.forEach(h => hx.insertAdjacentHTML("beforeend", `<label><input type="checkbox" name="${h}">${h.slice(3).replace(/_/g," ")}</label>`));
const P = [
 {age:78,sex:"F",arrival:"AMBULANCE",temperature:38.6,heartrate:112,resprate:24,o2sat:93,sbp:104,dbp:62,pain:3,hour:2,complaint:"Fever, cough",admissions:1,hx_diabetes:1,hx_chronic_lung:1},
 {age:24,sex:"M",arrival:"WALK IN",temperature:36.8,heartrate:78,resprate:16,o2sat:99,sbp:128,dbp:76,pain:6,hour:15,complaint:"R ankle injury",weekend:1},
 {age:84,sex:"M",arrival:"AMBULANCE",temperature:36.1,heartrate:104,resprate:22,o2sat:95,sbp:92,dbp:54,pain:0,hour:9,complaint:"Altered mental status",ed_visits:2,hx_kidney:1,hx_dementia:1}];
const f = document.getElementById("f"), out = document.getElementById("out");
document.querySelectorAll("[data-p]").forEach(b => b.onclick = () => {
  f.reset(); const p = P[+b.dataset.p];
  for (const [k,v] of Object.entries(p)) { const el = f.elements[k]; if (!el) continue; el.type==="checkbox" ? el.checked=!!v : el.value=v; }
  f.requestSubmit();
});
const pct = v => (v*100).toFixed(v < 0.1 ? 1 : 0) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
f.onsubmit = async e => {
  e.preventDefault();
  const data = {}; new FormData(f).forEach((v,k) => data[k] = v);
  const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)});
  const j = await r.json();
  if (j.error) { out.innerHTML = `<p>${esc(j.error)}</p>`; return; }
  const lamps = ["low","moderate","high"].map(t => `<span class="lamp ${t} ${t===j.tier?"on":""}" aria-hidden="true"></span>`).join("");
  const icon = {low:"✓", moderate:"!", high:"⚠"}[j.tier];
  const rs = j.reasons.map(x => `<li><b>${esc(x.label)}</b>${x.value===null?" (not recorded)":" = "+(+x.value.toFixed(2))} <span class="${x.contribution>0?"up":"down"}">${x.contribution>0?"↑":"↓"} ${x.effect}</span></li>`).join("");
  const ctx = j.tier_sepsis_rate != null ? `<p class="note">In the held-out test set, this tier held ${pct(j.tier_share_of_arrivals)} of arrivals and ${pct(j.tier_sepsis_rate)} of them had sepsis recorded.</p>` : "";
  const miss = j.set_to_missing.length ? `<p class="note">Outside plausible range, treated as not recorded: ${j.set_to_missing.join(", ")}.</p>` : "";
  out.innerHTML = `<div class="light">${lamps}</div><p class="tier">${icon} ${j.tier[0].toUpperCase()+j.tier.slice(1)} risk</p>
   <p class="risk">Estimated sepsis risk ${j.risk >= 0.95 ? "above 95%" : pct(j.risk)}</p><b>Top reasons</b><ol>${rs}</ol>
   <p class="note">Complaint read as: ${j.complaint_groups.map(esc).join(", ")}.</p>${miss}${ctx}
   <p class="note">Tiers: high = top 5% of arrivals by model score, moderate = next 15%. Research prototype, not for clinical use.</p>`;
};
</script></body></html>"""


def serve(app, port):
    page = PAGE.replace("__HX__", json.dumps([f for f in app.fm.features if f.startswith("hx_")])).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, page, "text/html; charset=utf-8") if self.path in ("/", "/index.html") else \
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b"not found", "text/plain")
            try:
                form = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                body = app.predict(form)
            except Exception as e:  # show the problem in the page instead of dropping the connection
                body = {"error": f"Could not score this patient: {e}"}
            self._send(200, json.dumps(body).encode(), "application/json")

        def log_message(self, *a):
            pass

    print(f"Traffic-light prototype on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parents[1] / "out" / "models" / "bundle.joblib"))
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    if not Path(a.bundle).exists():
        sys.exit(f"No models at {a.bundle}. Run `python run.py` first (or pass --bundle).")
    serve(App(a.bundle), a.port)
