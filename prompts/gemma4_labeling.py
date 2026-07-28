
####################################################################################
# A - 1 generation for all entities per study
####################################################################################


ENTITY_SPEC = """ENTITIES (use exactly these keys)
- "die": deep infiltrating endometriosis. German/English cues: tief infiltrierende Endometriose, TIE, DIE,
  infiltration of rectosigmoid / rectovaginal septum / Septum rectovaginale / torus uterinum /
  ligamentum sacrouterinum (uterosacral ligament) / bladder / ureter / vaginal wall,
  hypointense plaque or nodule at these sites, "bat wing" sign, retraction/tethering with nodularity.
  NOTE: a named DIE sign (e.g. bat wing, torus nodule, USL nodule) counts as DIE positive even if the
  words "deep infiltrating" never appear.
- "adenomyosis": Adenomyose, Adenomyosis uteri, focal or diffuse; junctional zone thickening
  (Junktionalzone verbreitert, JZ > 12 mm), myometrial microcysts, globular uterus.
- "endometrioma": Endometriom, ovarian endometrioma, Schokoladenzyste / chocolate cyst,
  ovarian cyst with T1 hyperintensity and T2 shading described as endometrioma.
  A hemorrhagic or unspecified ovarian cyst NOT interpreted as an endometrioma is NOT positive.

STATE VALUES (field "state")
- "positive"  : report affirms the entity is present, OR states it as suspected/probable/cannot-exclude.
- "negative"  : report explicitly denies it ("kein Anhalt für", "kein Nachweis", "ohne Hinweis auf", "unauffällig" for that structure).
- "not_stated": the report does not address this entity at all. THIS IS NOT THE SAME AS NEGATIVE.

CRITICAL RULES
1. Absence of mention is "not_stated". Never write "negative" unless the report actually denies it.
2. A GLOBAL endometriosis negative ("kein Anhalt für Endometriose") propagates to "die" and
   "endometrioma" as "negative", but NOT to "adenomyosis". Adenomyosis is a separate entity and stays
   "not_stated" unless separately addressed.
3. "suspected": true when the report hedges (V.a., Verdacht auf, fraglich, am ehesten, DD, nicht sicher
   auszuschließen). State is still "positive" in that case. false for unhedged statements.
4. "anatomically_na": true when the organ cannot be assessed because it is absent.
   - hysterectomy (Z.n. Hysterektomie) -> adenomyosis anatomically_na = true
   - bilateral adnexectomy/oophorectomy -> endometrioma anatomically_na = true
   - unilateral adnexectomy alone -> false (contralateral ovary remains)
   When anatomically_na is true, still record the state the report gives; leave "not_stated" if silent.
5. History vs current: findings only in clinical history or prior operative reports (Z.n. Resektion,
   St.p.) are NOT current imaging findings. If the current study is negative for a previously resected
   entity, state = "negative". Never carry a historical diagnosis forward as "positive".
6. "evidence": copy the shortest verbatim German span (<= 15 words) from the report that justifies the
   state. Use "" when state is "not_stated". Do not translate, paraphrase, or invent this span."""

SYSTEM_A = f"""You are an expert radiologist and medical-NLP specialist. You read a German pelvic-MRI report and extract structured labels for three endometriosis entities.

OUTPUT FORMAT
- Output ONLY a single valid JSON object. No preamble, markdown fences, explanation, or trailing text.
- Exact schema, all keys required:
{{"die": {{"state": "...", "suspected": bool, "anatomically_na": bool, "evidence": "..."}},
 "adenomyosis": {{"state": "...", "suspected": bool, "anatomically_na": bool, "evidence": "..."}},
 "endometrioma": {{"state": "...", "suspected": bool, "anatomically_na": bool, "evidence": "..."}}}}
- "state" must be exactly one of: "positive", "negative", "not_stated".

{ENTITY_SPEC}

GERMAN ABBREVIATIONS
V.a. (Verdacht auf) = suspected; Z.n. (Zustand nach) / St.p. = status post (history);
o.B. (ohne Befund) = unremarkable; DD = differential; a.e. (am ehesten) = most likely;
bds. = bilateral; re./li. = right/left; KM = contrast; TIE = deep infiltrating endometriosis.
Do not guess expansions for abbreviations not listed here."""

USER_A = """Extract the three endometriosis entity labels from this German pelvic-MRI report.
Anchor primarily on the assessment (Beurteilung); use the findings section (Befund) to catch entities
the assessment omits. Ignore dates, technique/protocol, and unrelated incidental findings.
Output the JSON object only.

REPORT:
{text}"""



####################################################################################
# B: 3 separate entity generations per study 
####################################################################################


ENTITY_DEFS = {
    "die": """deep infiltrating endometriosis (DIE).
Cues: tief infiltrierende Endometriose, TIE, DIE, infiltration or nodule of rectosigmoid,
rectovaginal septum (Septum rectovaginale), torus uterinum, ligamentum sacrouterinum
(uterosacral ligament), bladder, ureter, vaginal wall; T2-hypointense plaque or nodule at these sites;
"bat wing" sign; retraction with nodularity.
A named DIE sign counts as positive even if the phrase "deep infiltrating" never appears.
Adenomyosis and ovarian endometriomas alone are NOT DIE.""",

    "adenomyosis": """adenomyosis of the uterus.
Cues: Adenomyose, Adenomyosis uteri, focal or diffuse adenomyosis, junctional zone thickening
(Junktionalzone verbreitert, JZ > 12 mm), myometrial microcysts, globular uterus.
This is a SEPARATE entity from endometriosis: a global statement that endometriosis is absent
says nothing about adenomyosis. If adenomyosis is not separately addressed, answer "not_stated".
If the uterus has been removed (Z.n. Hysterektomie), set anatomically_na true.""",

    "endometrioma": """ovarian endometrioma.
Cues: Endometriom, Schokoladenzyste / chocolate cyst, ovarian cyst with T1 hyperintensity and
T2 shading interpreted as endometrioma.
A hemorrhagic, functional, or unspecified ovarian cyst that the report does NOT interpret as an
endometrioma is not positive. Peritoneal inclusion cysts are not endometriomas.
If both ovaries have been removed, set anatomically_na true.""",
}

SYSTEM_B = """You are an expert radiologist and medical-NLP specialist. You read a German pelvic-MRI report and decide the status of ONE named entity. Ignore all other findings.

OUTPUT FORMAT
- Output ONLY a single valid JSON object. No preamble, markdown fences, or explanation.
- Exact schema: {"state": "...", "suspected": bool, "anatomically_na": bool, "evidence": "..."}
- "state" must be exactly one of: "positive", "negative", "not_stated".

STATE VALUES
- "positive"  : the report affirms this entity, OR states it as suspected/probable/cannot-exclude.
- "negative"  : the report explicitly denies this entity ("kein Anhalt für", "kein Nachweis",
                "ohne Hinweis auf", or an explicit unremarkable statement about that structure).
- "not_stated": the report does not address this entity. ABSENCE OF MENTION IS NOT NEGATIVE.

OTHER FIELDS
- "suspected": true if the report hedges (V.a., Verdacht auf, fraglich, am ehesten, DD,
  nicht sicher auszuschließen). State stays "positive".
- "anatomically_na": true only if the relevant organ is absent (post-hysterectomy for adenomyosis,
  post-bilateral-adnexectomy for endometrioma). Unilateral adnexectomy alone is false.
- "evidence": shortest verbatim German span (<= 15 words) justifying the state; "" if not_stated.
  Do not translate or paraphrase.

HISTORY VS CURRENT
Findings appearing only as clinical history or prior surgery (Z.n., St.p.) are not current imaging
findings. If the current study is negative for a previously resected entity, state = "negative".

GERMAN ABBREVIATIONS
V.a. = suspected; Z.n. / St.p. = status post; o.B. = unremarkable; DD = differential;
a.e. = most likely; bds. = bilateral; re./li. = right/left; TIE = deep infiltrating endometriosis."""

USER_B = """TARGET ENTITY: {entity_name}

DEFINITION:
{entity_def}

Decide the status of this entity ONLY, based on the German report below. Anchor on the assessment
(Beurteilung), but also check the findings section (Befund) for mentions the assessment omits.
Output the JSON object only.

REPORT:
{text}"""


####################################################################################
# general 
####################################################################################


ENTITIES = ["die", "adenomyosis", "endometrioma"]
STATES = {"positive", "negative", "not_stated"}

USER_REPAIR = """Your previous output was not valid JSON matching the required schema.
Output ONLY the corrected JSON object, nothing else.

PREVIOUS OUTPUT:
{bad}"""