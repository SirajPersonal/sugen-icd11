"""Write eval/extra_notes.jsonl: a hand-made out-of-distribution check set.

The hidden generalisation slice covers "every chapter of ICD-11" and notes
"not always in English". The dev set has neither, so this file adds 45 short
notes I wrote myself: other chapters (neoplasms, mental health, nervous
system, eye, ear, injuries, neonatal, symptoms, special-purpose codes) and
other languages (French, Spanish, Portuguese, Swahili, Hindi, Arabic, Chinese).
Every gold code is checked against the MMS release when this script runs.

Caveat: these are written by the same person who wrote the lexicon, so they
are easier than real notes. They catch breakage and blind spots; the absolute
score is not a forecast.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coder.mms import Mms  # noqa: E402

NOTES = [
    # --- other chapters, English ---------------------------------------------------
    ("x-onc-01", "52F breast lump R upper outer quadrant x2/12, hard, irregular, axillary node palpable. Core biopsy: invasive ductal carcinoma, ER+. Imp: carcinoma of right breast. Refer oncology MDT.", ["2C61.0"]),
    ("x-onc-02", "68M smoker 40 pack-yrs, haemoptysis + wt loss. CXR mass R hilum, CT staging done at referral hospital: bronchial carcinoma, histology pending. Dx: lung cancer. Palliative review.", ["2C25.Z"]),
    ("x-onc-03", "71M on review. Known adenocarcinoma of prostate on ADT, PSA stable. Imp: carcinoma prostate, stable. Continue goserelin.", ["2C82.Z", "2C82.0"]),
    ("x-psy-01", "Doctor: How has your mood been?\nPatient: Low for about two months. I can't enjoy anything, I wake at 4am, I've lost weight, I'm struggling at work.\nDoctor: Any thoughts of harming yourself?\nPatient: No.\nDoctor: Has this ever happened before?\nPatient: Never.\nDoctor: This sounds like a moderate depressive episode, the first one you've had, no psychotic features. I'll start sertraline 50 milligrams and refer you for counselling.", ["6A70.1"]),
    ("x-psy-02", "34F constant worry about everything x 1 yr, restless, poor sleep, muscle tension, irritability. No panic attacks. GAD-7 16. Imp: Generalised anxiety disorder. CBT referral, sertraline.", ["6B00"]),
    ("x-psy-03", "9M, teacher reports daydreaming, losing things, not finishing tasks, no hyperactivity. Symptoms >6/12 at home and school. Imp: ADHD, predominantly inattentive. Parent training, school plan.", ["6A05.0"]),
    ("x-psy-04", "45M drinks 1 bottle spirits/day x5 yrs, morning shakes, failed attempts to cut down, drinking despite liver warning. Imp: alcohol dependence. Thiamine, brief intervention, refer rehab.", ["6C40.2Z", "6C40.20"]),
    ("x-slp-01", "29F trouble falling asleep x3 wks since new job, wakes tired. No mood symptoms. Dx: short-term insomnia. Sleep hygiene.", ["7A01"]),
    ("x-neu-01", "26F recurrent unilateral throbbing headaches with nausea and photophobia, 4-72h, no aura, 3/month. Normal neuro exam. Dx: migraine without aura. Sumatriptan PRN.", ["8A80.0"]),
    ("x-neu-02", "Doctor: When did you notice your face?\nPatient: This morning, I couldn't close my right eye and my mouth is drooping.\nDoctor: Can you raise your eyebrows? The whole right side of your face is weak, including the forehead. No weakness in your arms or legs, speech is fine. This is Bell's palsy. We'll start prednisolone and you need to tape the eye at night.", ["8B88.0"]),
    ("x-neu-03", "18m child, fever 39.5 from viral URTI, one generalised seizure lasting 2 min, recovered fully, no focal signs, first episode. Imp: simple febrile seizure. Antipyretics, parent education.", ["8A63.00"]),
    ("x-neu-04", "50F nocturnal tingling thumb, index and middle finger R hand, Phalen +, Tinel +. Dx: carpal tunnel syndrome. Wrist splint.", ["8C10.0"]),
    ("x-eye-01", "4F both eyes red, sticky yellow discharge, lids stuck in the morning. VA normal. Dx: bacterial conjunctivitis. Chloramphenicol drops.", ["9A60.3Z", "9A60.Z"]),
    ("x-eye-02", "74M gradual blurring both eyes x1yr, glare at night. Lens opacities bilat, VA 6/36. Imp: age-related cataract. Refer for surgery.", ["9B10.0Z"]),
    ("x-eye-03", "Painful lump on upper eyelid margin x2/7, red, pointing. Dx: stye. Warm compresses.", ["9A01.2Z", "9A01.20"]),
    ("x-ear-01", "3M ear pulling, fever 38.4, crying at night after a cold. Otoscopy: L TM red and bulging, no perforation. Dx: acute otitis media. Amoxicillin.", ["AB00"]),
    ("x-ear-02", "Swimmer, R ear pain and itch, tragal tenderness, canal oedematous with debris, TM normal. Dx: otitis externa. Ciprofloxacin ear drops, keep dry.", ["AA3Z"]),
    ("x-ren-01", "63M T2DM and HTN, eGFR 52 and 49 three months apart, ACR 4. Imp: CKD stage 3a. ACE-i, statin, annual review.", ["GB61.2"]),
    ("x-met-01", "48M routine check, fasting total cholesterol 7.2, LDL 5.1, TG normal. No secondary cause found. Dx: hypercholesterolaemia. Atorvastatin, diet.", ["5C80.0Z", "5C80.00"]),
    ("x-msk-01", "41F symmetrical small-joint polyarthritis hands x4/12, morning stiffness 2h. RF and anti-CCP positive. Dx: seropositive rheumatoid arthritis. Start methotrexate, refer rheumatology.", ["FA20.0"]),
    ("x-msk-02", "67F, menopause age 50, DEXA T-score -3.0 at hip, no fracture. Dx: postmenopausal osteoporosis. Alendronate, calcium + vit D.", ["FB83.11"]),
    ("x-skn-01", "Well-demarcated thick silvery scaly plaques on elbows and knees x3 yrs. Nails pitted. Dx: plaque psoriasis. Topical steroid + calcipotriol.", ["EA90.0"]),
    ("x-skn-02", "6F itchy dry skin flexures of elbows and knees since infancy, lichenified, family hx asthma. Dx: atopic eczema (childhood). Emollients, hydrocortisone 1%.", ["EA80.1", "EA80.Z"]),
    ("x-skn-03", "Doctor: Where is the rash?\nPatient: Just here around my left side, it started with burning pain three days ago.\nDoctor: I see clusters of blisters in a band on the left chest, not crossing the middle. Your eye is fine. This is shingles, no complications. I'll give you acyclovir and pain relief.", ["1E91.0"]),
    ("x-inf-01", "Fever 3/7, retro-orbital pain, myalgia, rash. NS1 positive. No abdominal pain, no bleeding, no vomiting, platelets 140. Dx: dengue without warning signs. Oral fluids, paracetamol, daily FBC.", ["1D20"]),
    ("x-inf-02", "Sore throat, fever 38.8, tonsillar exudate, tender anterior nodes, no cough. Rapid strep antigen POSITIVE. Dx: strep throat. Penicillin V x10/7.", ["1B51"]),
    ("x-inf-03", "Cough, fever, anosmia x4/7. SARS-CoV-2 PCR positive. SpO2 97% RA, chest clear. Dx: COVID-19, mild. Isolation, paracetamol.", ["RA01.0"]),
    ("x-hem-01", "14M known SCD (HbSS), severe pain both legs and back x1/7, no fever, no chest symptoms. Imp: vaso-occlusive crisis. Analgesia, fluids, admit.", ["3A51.2"]),
    ("x-gi-01", "22M periumbilical pain migrated to RIF x18h, anorexia, T 37.9, Rovsing +, no guarding or rigidity. Imp: acute appendicitis, no peritonitis. Refer surgical.", ["DB10.02"]),
    ("x-gi-02", "Reducible swelling R groin, expansile cough impulse, above and medial to pubic tubercle. Dx: inguinal hernia. Elective surgical referral.", ["DD51"]),
    ("x-den-01", "Toothache lower right molar on sweets, visible cavity, no swelling. Dx: dental caries. Refer for filling.", ["DA08.0"]),
    ("x-inj-01", "Inversion injury L ankle playing football, swelling lateral malleolus, can weight bear 4 steps, Ottawa negative. Dx: ankle sprain. RICE, analgesia.", ["ND14.7Z", "ND14.73"]),
    ("x-inj-02", "Fall on outstretched hand, dinner-fork deformity R wrist. X-ray: distal radius fracture with dorsal angulation. Dx: Colles fracture. POP, ortho referral.", ["NC32.50", "NC32.5Z"]),
    ("x-inj-03", "Hot tea spilled on L hand. Erythema with blisters on dorsum, painful, 1% TBSA. Dx: superficial partial thickness burn of hand. Dressing, analgesia.", ["ND95.1"]),
    ("x-neo-01", "Day 4 term baby, breastfeeding well, yellow to chest, TSB 210, below phototherapy line, mother O+ baby O+. Imp: neonatal jaundice. Review TSB in 24h.", ["KA87.Z", "KA87.Y"]),
    # --- other languages -----------------------------------------------------------
    ("x-fr-01", "Homme 45 ans, fièvre 39°, toux productive et douleur thoracique droite depuis 4 jours. Crépitants base droite. Radio : opacité lobaire inférieure droite. Diagnostic : pneumonie communautaire. Amoxicilline 1 g x3/j.", ["CA40.Z"]),
    ("x-fr-02", "Enfant 6 ans, fièvre, frissons, TDR paludisme positif (P. falciparum). Conscient, pas de convulsions, pas d'anémie sévère. Diagnostic : paludisme simple. Artéméther-luméfantrine.", ["1F40.Z"]),
    ("x-es-01", "Mujer 28 años, embarazo de 20 semanas, disuria y polaquiuria, sin fiebre ni dolor lumbar. Tira reactiva: nitritos positivos. Diagnóstico: cistitis en el embarazo. Nitrofurantoína.", ["JA62.1"]),
    ("x-es-02", "Varón 60 años, PA 158/96 en tres visitas, sin daño de órgano. Diagnóstico: hipertensión arterial esencial. Inicia amlodipino.", ["BA00.Z", "BA00.0"]),
    ("x-pt-01", "Paciente 35 anos, tosse há 3 semanas, sudorese noturna, perda de peso. Baciloscopia positiva. Diagnóstico: tuberculose pulmonar confirmada. Inicia esquema RHZE.", ["1B10.0"]),
    ("x-sw-01", "Mtoto wa miaka 2, kuhara mara 6 kwa siku kwa siku 2, kutapika, homa kidogo. Macho yamezama kidogo. Utambuzi: gastroenteritis ya kuambukiza. ORS na zinki.", ["1A40.Z"]),
    ("x-sw-02", "Mwanamume miaka 50, kiu sana, kukojoa mara kwa mara, sukari ya damu (FBS) 11.2 mara mbili. Utambuzi: kisukari aina ya pili. Metformin.", ["5A11"]),
    ("x-hi-01", "रोगी 55 वर्ष, पुरुष। प्यास ज़्यादा, बार-बार पेशाब, HbA1c 8.4%। निदान: टाइप 2 मधुमेह। मेटफॉर्मिन 500 mg दिन में दो बार।", ["5A11"]),
    ("x-ar-01", "مريضة ٢٤ سنة، معروفة بالربو منذ الطفولة، ضيق تنفس وأزيز منذ أمس رغم البخاخ. التشخيص: نوبة ربو متوسطة. سالبوتامول بالرذاذ وبريدنيزولون.", ["CA23.30"]),
    ("x-zh-01", "男，62岁。头晕，多次测血压 165/100 mmHg，无继发性高血压证据。诊断：原发性高血压。氨氯地平 5mg 每日一次。", ["BA00.Z", "BA00.0"]),
]


def main():
    mms = Mms()
    bad = [(i, g) for i, _, gs in NOTES for g in gs if g not in mms.cats]
    if bad:
        sys.exit(f"gold codes not in MMS: {bad}")
    out = ROOT / "eval" / "extra_notes.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, text, gold in NOTES:
            f.write(json.dumps({"id": i, "text": text, "gold": gold, "slice": "B",
                                "style": "extra_" + i.split("-")[1]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(NOTES)} notes to {out}")


if __name__ == "__main__":
    main()
