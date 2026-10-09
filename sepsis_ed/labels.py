"""Sepsis-3 onset labels for ED stays (Singer et al., JAMA 2016), following the
MIMIC-code `sepsis3` concept: suspected infection + SOFA >= 2 around it.

Suspected infection (t_si):
  * culture drawn <=72h BEFORE an antibiotic  -> t_si = culture time
  * culture drawn <=24h AFTER an antibiotic   -> t_si = antibiotic time
Organ dysfunction: worst value of each SOFA component in [t_si-48h, t_si+24h],
summed, baseline assumed 0. Sepsis onset = t_si when that sum >= 2.
A stay is positive if onset falls in [ED intime, intime + horizon_h].

Known simplifications (documented in README): window-worst SOFA rather than
hourly; no urine output for renal; respiratory 3-4 points not gated on
ventilation; vasopressors scored on norepinephrine-equivalent dose
(any pressor -> 3, >0.1 mcg/kg/min -> 4) rather than per-drug SOFA bands.
"""
import re

import numpy as np
import pandas as pd

from .io import exists, hours, read

ABX = """adoxa ala-tet alodox amikacin amoxicillin ampicillin augmentin avelox avidoxy azactam
azithromycin aztreonam axetil bactocill bactrim bethkis biaxin bicillin cayston cefazolin cedax
cefoxitin ceftazidime cefaclor cefadroxil cefdinir cefditoren cefepime cefotan cefotetan cefotaxime
ceftaroline cefpodoxime cefprozil ceftibuten ceftin ceftriaxone cefuroxime cephalexin cephalothin
cephapirin chloramphenicol cipro ciprofloxacin claforan clarithromycin cleocin clindamycin cubicin
daptomycin dicloxacillin doryx doxycy duricef dynacin ertapenem ery-tab eryped eryc erythrocin
erythromycin factive flagyl fortaz furadantin garamycin gentamicin imipenem kanamycin keflex kefzol
ketek levaquin levofloxacin lincocin linezolid macrobid macrodantin maxipime meropenem metronidazole
minocin minocycline monodox monurol moxifloxacin nafcillin nitrofurantoin norfloxacin ofloxacin
omnicef oxacillin penicillin pfizerpen piperacillin rifampin rocephin septra solodyn spectracef
streptomycin sulfamethoxazole sulfatrim tazicef tetracycline timentin tobramycin trimethoprim unasyn
vancocin vancomycin vantin vibativ vibramycin zinacef zithromax zosyn zyvox""".split()
ABX_RE = "|".join(re.escape(a) for a in ABX)
# First DataBank GSN codes for systemic antibiotics, from MIMIC-Sepsis (Huang et al. 2025,
# github.com/yongh7/MIMIC-sepsis, MIT), originally AI_Clinician (Komorowski et al. 2018).
ABX_GSN = set("""002542 002543 007371 008873 008877 008879 008880 008935 008941 008942 008943 008944
008983 008984 008990 008991 008992 008995 008996 008998 009043 009046 009065 009066 009136 009137 009162
009164 009165 009171 009182 009189 009213 009214 009218 009219 009221 009226 009227 009235 009242 009263
009273 009284 009298 009299 009310 009322 009323 009326 009327 009339 009346 009351 009354 009362 009394
009395 009396 009509 009510 009511 009544 009585 009591 009592 009630 013023 013645 013723 013724 013725
014182 014500 015979 016368 016373 016408 016931 016932 016949 018636 018637 018766 019283 021187 021205
021735 021871 023372 023989 024095 024194 024668 025080 026721 027252 027465 027470 029325 029927 029928
037042 039551 039806 040819 041798 043350 043879 044143 045131 045132 046771 047797 048077 048262 048266
048292 049835 050442 050443 051932 052050 060365 066295 067471""".split())
# ICU culture orders charted in chartevents (blood, urine, CSF, sputum, line tips, BAL, ...), same source
CHART_CULTURE = [225401, 225437, 225444, 225451, 225454, 225814, 225816, 225817, 225818, 225722, 225723,
                 225724, 225725, 225726, 225727, 225728, 225729, 225730, 225731, 225732, 225733, 227726,
                 225734, 225735, 225736, 225768, 226131]
NON_SYSTEMIC_RE = r"cream|gel|ophth|oint|desensitization|drops|lotion|topical"
NON_SYSTEMIC_ROUTES = {"OU", "OS", "OD", "AU", "AS", "AD", "TP", "EAR", "EYE", "TOP"}

# itemids
LAB = {"platelets": 51265, "bilirubin": 50885, "creatinine": 50912, "pao2": 50821, "fio2_lab": 50816}
CHART_MAP = [220052, 220181, 225312]
CHART_FIO2 = 223835
CHART_GCS = {220739: "eye", 223900: "verbal", 223901: "motor"}
VASO = {221906: "norepinephrine", 221289: "epinephrine", 221662: "dopamine",
        221653: "dobutamine", 222315: "vasopressin", 221749: "phenylephrine"}


# ---------------------------------------------------------------- infection
def _gsn_match(gsn):
    """prescriptions.gsn can hold several space-separated codes; match if any is an antibiotic."""
    g = gsn.fillna("").astype(str).str.split()
    return g.apply(lambda codes: any(c.zfill(6) in ABX_GSN for c in codes))


def antibiotics(mimic, ed):
    """Systemic antibiotic = GSN code match OR drug-name match, minus topical/ophthalmic forms."""
    rx = read(mimic, "hosp", "prescriptions", ["subject_id", "starttime", "drug", "route", "gsn"], ["starttime"],
              dtype={"gsn": str})
    d = rx.drug.fillna("").str.lower()
    keep = (_gsn_match(rx.gsn) | d.str.contains(ABX_RE)) & ~d.str.contains(NON_SYSTEMIC_RE) & \
        ~rx.route.fillna("").str.upper().isin(NON_SYSTEMIC_ROUTES)
    out = [rx.loc[keep, ["subject_id", "starttime"]].rename(columns={"starttime": "abx_time"})]
    if exists(ed, "ed", "pyxis"):  # antibiotics dispensed in the ED itself
        px = read(ed, "ed", "pyxis", ["subject_id", "charttime", "name", "gsn"], ["charttime"], dtype={"gsn": str})
        n = px.name.fillna("").str.lower()
        keep = (_gsn_match(px.gsn) | n.str.contains(ABX_RE)) & ~n.str.contains(NON_SYSTEMIC_RE)
        out.append(px.loc[keep, ["subject_id", "charttime"]].rename(columns={"charttime": "abx_time"}))
    return pd.concat(out).dropna().drop_duplicates()


def cultures(mimic, ce):
    """Cultures from hosp microbiology plus ICU culture orders charted in chartevents."""
    mb = read(mimic, "hosp", "microbiologyevents", ["subject_id", "chartdate", "charttime"],
              ["chartdate", "charttime"])
    mb["culture_time"] = mb.charttime.fillna(mb.chartdate)
    icu = ce.loc[ce.itemid.isin(CHART_CULTURE), ["subject_id", "charttime"]].rename(
        columns={"charttime": "culture_time"})
    return pd.concat([mb[["subject_id", "culture_time"]], icu]).dropna().drop_duplicates()


def load_chartevents(mimic):
    """One pass over chartevents for every itemid the labels need (it is ~30 GB in the full release)."""
    cols = ["subject_id", "itemid", "charttime", "valuenum"]
    if not exists(mimic, "icu", "chartevents"):
        return pd.DataFrame(columns=cols).astype({"charttime": "datetime64[ns]"})
    want = set(CHART_MAP + [CHART_FIO2] + list(CHART_GCS) + CHART_CULTURE)
    return read(mimic, "icu", "chartevents", cols, ["charttime"], filter_fn=lambda c: c.itemid.isin(want))


def suspected_infection(abx, cx):
    m = abx.merge(cx, on="subject_id")
    d = hours(m.culture_time - m.abx_time)
    before = m[(d <= 0) & (d >= -72)].assign(si_time=lambda x: x.culture_time)
    after = m[(d > 0) & (d <= 24)].assign(si_time=lambda x: x.abx_time)
    return pd.concat([before, after])[["subject_id", "si_time"]].drop_duplicates()


# ---------------------------------------------------------------- SOFA
def _obs(df, time, component, score):
    return pd.DataFrame({"subject_id": df.subject_id.values, "time": df[time].values,
                         "component": component, "score": np.asarray(score, dtype=int)})


def _band(x, cuts, high_is_bad=True):
    """cuts = thresholds for scores 1..4."""
    s = np.zeros(len(x), dtype=int)
    for i, c in enumerate(cuts, start=1):
        s = np.where((x >= c) if high_is_bad else (x < c), i, s)
    return s


def norepinephrine_equivalent(ie):
    """Vasopressor rate -> norepinephrine-equivalent mcg/kg/min. Unit handling follows MIMIC-Sepsis
    vaso.py; factors follow Goradia et al., J Crit Care 2021 (epi 1, phenylephrine 0.1, dopamine 0.01,
    vasopressin 2.5 per U/min). MIMIC-Sepsis uses 0.45 and 5 for the last two, which overstates dose.
    Uses charted weight, 80 kg when missing."""
    drug, uom = ie.itemid.map(VASO), ie.rateuom.fillna("")
    r = ie.rate.astype(float)
    w = ie.patientweight.where(ie.patientweight.between(20, 300), 80.0) if "patientweight" in ie else 80.0
    per_kg = np.where(uom.str.contains("/kg/"), r, np.where(uom.str.contains("mcg/min"), r / w, np.nan))
    vaso_u_min = np.where(uom == "units/min", r, np.where(uom == "units/hour", r / 60, np.nan))
    return pd.Series(np.select(
        [drug.isin(["norepinephrine", "epinephrine"]), drug == "phenylephrine", drug == "dopamine",
         drug == "vasopressin"],
        [per_kg, per_kg * 0.1, per_kg * 0.01, vaso_u_min * 2.5], default=0.0), index=ie.index).fillna(0)


def sofa_observations(mimic, ed, ce):
    obs = []
    ce = ce.dropna(subset=["valuenum"])
    lab = read(mimic, "hosp", "labevents", ["subject_id", "itemid", "charttime", "valuenum"], ["charttime"],
               filter_fn=lambda c: c.itemid.isin(LAB.values())).dropna(subset=["valuenum"])
    g = {k: lab[lab.itemid == v] for k, v in LAB.items()}
    obs.append(_obs(g["platelets"], "charttime", "coag", _band(g["platelets"].valuenum, [150, 100, 50, 20], False)))
    obs.append(_obs(g["bilirubin"], "charttime", "liver", _band(g["bilirubin"].valuenum, [1.2, 2, 6, 12])))
    obs.append(_obs(g["creatinine"], "charttime", "renal", _band(g["creatinine"].valuenum, [1.2, 2, 3.5, 5])))

    # respiratory: PaO2 / most recent FiO2 within 4h (lab or charted)
    fio2 = pd.concat([g["fio2_lab"], ce[ce.itemid == CHART_FIO2]])[["subject_id", "charttime", "valuenum"]]
    fio2 = fio2.assign(fio2=lambda x: np.where(x.valuenum <= 1, x.valuenum * 100, x.valuenum))
    fio2 = fio2[(fio2.fio2 >= 21) & (fio2.fio2 <= 100)].sort_values("charttime")
    pao2 = g["pao2"].sort_values("charttime")
    if len(pao2) and len(fio2):
        pf = pd.merge_asof(pao2, fio2[["subject_id", "charttime", "fio2"]], on="charttime", by="subject_id",
                           direction="backward", tolerance=pd.Timedelta("4h")).dropna(subset=["fio2"])
        ratio = pf.valuenum / (pf.fio2 / 100)
        obs.append(_obs(pf, "charttime", "resp", _band(ratio, [400, 300, 200, 100], False)))

    # cardiovascular: MAP < 70 from ED vitals + ICU charting
    maps = []
    if exists(ed, "ed", "vitalsign"):
        vs = read(ed, "ed", "vitalsign", ["subject_id", "charttime", "sbp", "dbp"], ["charttime"])
        maps.append(vs.assign(map=(vs.sbp + 2 * vs.dbp) / 3)[["subject_id", "charttime", "map"]])
    maps.append(ce[ce.itemid.isin(CHART_MAP)].rename(columns={"valuenum": "map"})[["subject_id", "charttime", "map"]])
    mp = pd.concat(maps).dropna()
    mp = mp[mp["map"].between(10, 300)]
    obs.append(_obs(mp, "charttime", "cardio", (mp["map"] < 70).astype(int)))

    # cardiovascular: vasopressors
    if exists(mimic, "icu", "inputevents"):
        ie = read(mimic, "icu", "inputevents", ["subject_id", "itemid", "starttime", "endtime", "rate", "rateuom",
                                                "patientweight"],
                  ["starttime", "endtime"], filter_fn=lambda c: c.itemid.isin(VASO))
        ie["ne_eq"] = norepinephrine_equivalent(ie)
        drug = ie.itemid.map(VASO)
        # SOFA CV: NE-equivalent >0.1 mcg/kg/min -> 4, any pressor -> 3, dobutamine alone -> 2
        score = np.select([ie.ne_eq > 0.1, ie.ne_eq > 0, drug == "dobutamine"], [4, 3, 2], default=0)
        ie = ie.assign(score=score)[score > 0]
        for t in ("starttime", "endtime"):
            obs.append(_obs(ie, t, "cardio", ie.score))

    # CNS: GCS total
    gcs = ce[ce.itemid.isin(CHART_GCS)]
    if len(gcs):
        tot = gcs.pivot_table(index=["subject_id", "charttime"], columns="itemid", values="valuenum", aggfunc="min")
        tot = tot.dropna().sum(axis=1).rename("gcs").reset_index()
        obs.append(_obs(tot, "charttime", "cns", _band(tot.gcs, [15, 13, 10, 6], False)))

    o = pd.concat(obs, ignore_index=True)
    o["time"] = pd.to_datetime(o.time)
    return o.dropna(subset=["time"])


def sepsis3_events(si, obs):
    m = si.merge(obs, on="subject_id")
    d = hours(m.time - m.si_time)
    w = m[(d >= -48) & (d <= 24)]
    sofa = (w.groupby(["subject_id", "si_time", "component"]).score.max()
             .groupby(["subject_id", "si_time"]).sum().rename("sofa").reset_index())
    return sofa[sofa.sofa >= 2]


# ---------------------------------------------------------------- ICD label
ICD10 = ("A40", "A41", "R652")
ICD9 = ("038", "99591", "99592", "78552")


def _is_sepsis_icd(code, ver):
    code = code.astype(str).str.replace(".", "", regex=False).str.upper()
    return ((ver == 10) & code.str.startswith(ICD10)) | ((ver == 9) & code.str.startswith(ICD9))


# ---------------------------------------------------------------- per-stay labels
def build_labels(mimic, ed, horizon_h=48):
    stays = read(ed, "ed", "edstays", ["subject_id", "hadm_id", "stay_id", "intime", "outtime"], ["intime", "outtime"])

    ce = load_chartevents(mimic)
    si = suspected_infection(antibiotics(mimic, ed), cultures(mimic, ce))
    ev = sepsis3_events(si, sofa_observations(mimic, ed, ce))

    m = stays[["stay_id", "subject_id", "intime"]].merge(ev, on="subject_id")
    m = m[hours(m.si_time - m.intime).between(0, horizon_h)]
    first = m.sort_values("si_time").groupby("stay_id").first()[["si_time", "sofa"]]
    lab = stays.set_index("stay_id").join(first)
    lab["sepsis3"] = lab.si_time.notna().astype(int)
    lab["onset_hours_from_arrival"] = hours(lab.si_time - lab.intime)

    # secondary label: sepsis ICD code on ED stay or linked admission
    dx = read(mimic, "hosp", "diagnoses_icd", ["hadm_id", "icd_code", "icd_version"])
    hadm_pos = set(dx.loc[_is_sepsis_icd(dx.icd_code, dx.icd_version), "hadm_id"])
    ed_pos = set()
    if exists(ed, "ed", "diagnosis"):
        edx = read(ed, "ed", "diagnosis", ["stay_id", "icd_code", "icd_version"])
        ed_pos = set(edx.loc[_is_sepsis_icd(edx.icd_code, edx.icd_version), "stay_id"])
    lab["sepsis_icd"] = (lab.hadm_id.isin(hadm_pos) | lab.index.isin(ed_pos)).astype(int)
    lab["admitted"] = lab.hadm_id.notna().astype(int)
    return lab.reset_index()[["stay_id", "subject_id", "hadm_id", "intime", "admitted", "sepsis3",
                              "si_time", "sofa", "onset_hours_from_arrival", "sepsis_icd"]]
