from newborns_dashboard.dash_co_newborn import NEWBORN_KPI_MAPPING, NEWBORN_KPI_OPTIONS


def validate_newborn_indicator(kpi_name: str):
    """Return (True, None) if KPI is valid for newborn; otherwise (False, msg)."""
    if kpi_name in NEWBORN_KPI_MAPPING:
        return True, None
    return False, "That looks like a maternal indicator. Type `maternal` to switch programs, or say `list indicators` to see newborn options."


NEWBORN_KPI_ALIASES = {
    # Core volume
    "admitted newborns": "Admitted Newborns",
    "admitted newborn": "Admitted Newborns",
    "admitted babies": "Admitted Newborns",
    "admitted neonates": "Admitted Newborns",
    "newborn admissions": "Admitted Newborns",
    "newborn admission": "Admitted Newborns",
    "total newborns": "Admitted Newborns",
    "total admitted newborns": "Admitted Newborns",
    "admittted newborns": "Admitted Newborns",
    "admittted newborn": "Admitted Newborns",
    "admitted newbron": "Admitted Newborns",
    "admitted newbonr": "Admitted Newborns",
    "admitted newbrn": "Admitted Newborns",
    "admitted newbrons": "Admitted Newborns",
    "admitted newborm": "Admitted Newborns",
    "admitted nebrn": "Admitted Newborns",
    "admitted nebrons": "Admitted Newborns",
    "admitted new borns": "Admitted Newborns",
    "admitted new born": "Admitted Newborns",
    "admitted newbown": "Admitted Newborns",
    "admitted newbworn": "Admitted Newborns",
    "admitted nb": "Admitted Newborns",
    "nb admissions": "Admitted Newborns",
    "nb admission": "Admitted Newborns",
    "new born admissions": "Admitted Newborns",
    "newbron admissions": "Admitted Newborns",
    "newbrn admissions": "Admitted Newborns",
    "newborn count": "Admitted Newborns",
    "newborn census": "Admitted Newborns",
    "newborn volume": "Admitted Newborns",
    "admit newborns": "Admitted Newborns",
    "admission newborns": "Admitted Newborns",
    "newborn admit": "Admitted Newborns",
    "newborn admission count": "Admitted Newborns",
    "newborn admits": "Admitted Newborns",
    "admitted nbabies": "Admitted Newborns",
    "neonate admissions": "Admitted Newborns",
    "neonates admitted": "Admitted Newborns",
    "neonatal admissions": "Admitted Newborns",
    "newbron count": "Admitted Newborns",
    "newbrons count": "Admitted Newborns",
    "newbron total": "Admitted Newborns",

    # Coverage rate
    "newborn coverage rate": "Newborn Coverage Rate",
    "newborn coverage": "Newborn Coverage Rate",
    "coverage rate": "Newborn Coverage Rate",
    "admission coverage": "Newborn Coverage Rate",
    "admissions coverage": "Newborn Coverage Rate",
    "newborn admission coverage": "Newborn Coverage Rate",
    "newborn admissions coverage": "Newborn Coverage Rate",
    "covergae rate": "Newborn Coverage Rate",
    "newborn covergae rate": "Newborn Coverage Rate",

    # Inborn / Outborn
    "inborn": "Inborn Rate (%)",
    "in born": "Inborn Rate (%)",
    "in-born": "Inborn Rate (%)",
    "facility born": "Inborn Rate (%)",
    "delivered here": "Inborn Rate (%)",
    "born here": "Inborn Rate (%)",
    "born inside": "Inborn Rate (%)",
    "internal birth": "Inborn Rate (%)",
    "in facility birth": "Inborn Rate (%)",
    "facility delivery": "Inborn Rate (%)",
    "inborn rate": "Inborn Rate (%)",
    "inborn babies": "Inborn Rate (%)",
    "outborn": "Outborn Rate (%)",
    "out born": "Outborn Rate (%)",
    "out-born": "Outborn Rate (%)",
    "referred newborns": "Outborn Rate (%)",
    "transferred newborns": "Outborn Rate (%)",
    "born outside": "Outborn Rate (%)",
    "home born": "Outborn Rate (%)",
    "external birth": "Outborn Rate (%)",
    "born elsewhere": "Outborn Rate (%)",
    "outside facility birth": "Outborn Rate (%)",
    "outside delivery": "Outborn Rate (%)",
    "out facility birth": "Outborn Rate (%)",
    "outborn rate": "Outborn Rate (%)",
    "outborn babies": "Outborn Rate (%)",

    # Hypothermia
    "hypothermia": "Hypothermia on Admission Rate (%)",
    "hypthermia": "Hypothermia on Admission Rate (%)",
    "hipothermia": "Hypothermia on Admission Rate (%)",
    "hypo thermia": "Hypothermia on Admission Rate (%)",
    "hypothrmia": "Hypothermia on Admission Rate (%)",
    "cold babies": "Hypothermia on Admission Rate (%)",
    "inborn hypothermia": "Inborn Hypothermia Rate (%)",
    "inborn hypothermia rate": "Inborn Hypothermia Rate (%)",
    "outborn hypothermia": "Outborn Hypothermia Rate (%)",
    "outborn hypothermia rate": "Outborn Hypothermia Rate (%)",
    "hypothermia inborn": "Inborn Hypothermia Rate (%)",
    "hypothermia outborn": "Outborn Hypothermia Rate (%)",
    "low temp": "Hypothermia on Admission Rate (%)",
    "low temperature": "Hypothermia on Admission Rate (%)",
    "temperature low": "Hypothermia on Admission Rate (%)",
    "hypothermic": "Hypothermia on Admission Rate (%)",
    "hypothermal": "Hypothermia on Admission Rate (%)",
    "cold newborns": "Hypothermia on Admission Rate (%)",
    "cold baby": "Hypothermia on Admission Rate (%)",
    "cold on arrival": "Hypothermia on Admission Rate (%)",
    "cold at admission": "Hypothermia on Admission Rate (%)",
    "low body temp": "Hypothermia on Admission Rate (%)",

    # Mortality
    "neonatal death": "Neonatal Mortality Rate (%)",
    "neonatal deaths": "Neonatal Mortality Rate (%)",
    "neonatal mortality": "Neonatal Mortality Rate (%)",
    "newborn death": "Neonatal Mortality Rate (%)",
    "newborn deaths": "Neonatal Mortality Rate (%)",
    "newborn mortality": "Neonatal Mortality Rate (%)",
    "mortality": "Neonatal Mortality Rate (%)",
    "nmr": "Neonatal Mortality Rate (%)",
    "nnmr": "Neonatal Mortality Rate (%)",
    "death rate": "Neonatal Mortality Rate (%)",
    "mortality rate": "Neonatal Mortality Rate (%)",
    "neonate death rate": "Neonatal Mortality Rate (%)",
    "neonate mortality": "Neonatal Mortality Rate (%)",
    "newborn fatality": "Neonatal Mortality Rate (%)",
    "newborn fatality rate": "Neonatal Mortality Rate (%)",

    # Birth weight & KMC
    "birth weight": "Birth Weight Rate",
    "birthweight": "Birth Weight Rate",
    "birht weight": "Birth Weight Rate",
    "bw": "Birth Weight Rate",
    "weight distribution": "Birth Weight Rate",
    "birth weight distribution": "Birth Weight Rate",
    "low birth weight share": "Birth Weight Rate",
    "lbw rate": "Birth Weight Rate",
    "vlbw rate": "Birth Weight Rate",
    "very low birth weight": "Birth Weight Rate",
    "low birth weight": "Birth Weight Rate",
    "birth wt": "Birth Weight Rate",
    "bwt": "Birth Weight Rate",
    "bw rate": "Birth Weight Rate",
    "bw distribution": "Birth Weight Rate",
    "bw breakdown": "Birth Weight Rate",
    "birth weight breakdown": "Birth Weight Rate",
    "kmc": "KMC Coverage by Birth Weight",
    "kangaroo": "KMC Coverage by Birth Weight",
    "kangaro": "KMC Coverage by Birth Weight",
    "kangaroo mother care": "KMC Coverage by Birth Weight",
    "skin to skin": "KMC Coverage by Birth Weight",
    "skin-to-skin": "KMC Coverage by Birth Weight",
    "kmc coverage": "KMC Coverage by Birth Weight",
    "kmc covergae": "KMC Coverage by Birth Weight",
    "k m c": "KMC Coverage by Birth Weight",
    "kangaroo care": "KMC Coverage by Birth Weight",
    "kmc by weight": "KMC Coverage by Birth Weight",
    "kmc coverage by weight": "KMC Coverage by Birth Weight",
    "kmc coverage rate": "KMC Coverage by Birth Weight",
    "kmc coverage percent": "KMC Coverage by Birth Weight",
    "kmc percent": "KMC Coverage by Birth Weight",

    # CPAP
    "cpap": "CPAP for RDS",
    "c pap": "CPAP for RDS",
    "c-pap": "CPAP for RDS",
    "rds": "CPAP for RDS",
    "respiratory distress": "CPAP for RDS",
    "cpap rds": "CPAP for RDS",
    "cpap for rds": "CPAP for RDS",
    "cpap by weight": "CPAP Coverage by Birth Weight",
    "cpap coverage by weight": "CPAP Coverage by Birth Weight",
    "cpap weight": "CPAP Coverage by Birth Weight",
    "ncpap": "CPAP for RDS",
    "nasal cpap": "CPAP for RDS",
    "breathing support": "CPAP for RDS",
    "resp support": "CPAP for RDS",
    "respiratory support": "CPAP for RDS",
    "cpap coverage": "CPAP Coverage by Birth Weight",
    "cpap coverage weight": "CPAP Coverage by Birth Weight",
    "cpap coverage by bw": "CPAP Coverage by Birth Weight",
    "cpap coverage %": "CPAP Coverage by Birth Weight",
    "cpap coverage rate": "CPAP Coverage by Birth Weight",
    "cpap percent": "CPAP Coverage by Birth Weight",
    "cpap percentage": "CPAP Coverage by Birth Weight",

    # Data quality (missing)
    "temperature": "Hypothermia on Admission Rate (%)",
    "missing temperature": "Missing Temperature (%)",
    "missing temp": "Missing Temperature (%)",
    "temp missing": "Missing Temperature (%)",
    "missing birth weight": "Missing Birth Weight (%)",
    "missing weight": "Missing Birth Weight (%)",
    "weight missing": "Missing Birth Weight (%)",
    "missing discharge status": "Missing Status of Discharge (%)",
    "missing status of discharge": "Missing Status of Discharge (%)",
    "missing newborn status": "Missing Status of Discharge (%)",
    "missing discharge outcome": "Missing Status of Discharge (%)",
    "status discharge missing": "Missing Status of Discharge (%)",
    "missing birth location": "Missing Birth Location (%)",
    "missing delivery location": "Missing Birth Location (%)",
    "missing place of birth": "Missing Birth Location (%)",
    "missing birth place": "Missing Birth Location (%)",
    "missing birth site": "Missing Birth Location (%)",
    "birth place missing": "Missing Birth Location (%)",
    "discharge status missing": "Missing Status of Discharge (%)",
    "status at discharge missing": "Missing Status of Discharge (%)",
    "discharge outcome missing": "Missing Status of Discharge (%)",
    "temperature missing": "Missing Temperature (%)",
    "birth weight missing": "Missing Birth Weight (%)",
    "missing admission temperature": "Missing Temperature (%)",
    "missing temp at admission": "Missing Temperature (%)",
    "missing temp admission": "Missing Temperature (%)",
    "no temperature": "Missing Temperature (%)",
    "no temp": "Missing Temperature (%)",
    "temp not recorded": "Missing Temperature (%)",
    "weight not recorded": "Missing Birth Weight (%)",
    "bw missing": "Missing Birth Weight (%)",
    "no discharge status": "Missing Status of Discharge (%)",
    "status not recorded": "Missing Status of Discharge (%)",
    "birth location not recorded": "Missing Birth Location (%)",
    "no birth location": "Missing Birth Location (%)",
    "missing birth weight rate": "Missing Birth Weight (%)",
    "missing birthweight rate": "Missing Birth Weight (%)",
    "missing bw rate": "Missing Birth Weight (%)",
    "missing weight rate": "Missing Birth Weight (%)",
    "missing birth wt": "Missing Birth Weight (%)",
    "missing birthweight": "Missing Birth Weight (%)",
    "missing bw": "Missing Birth Weight (%)",
    "missingbirthweight": "Missing Birth Weight (%)",
    "missingbirthweightrate": "Missing Birth Weight (%)",
    "missingbirthweight%": "Missing Birth Weight (%)",
    "missingbirthwt": "Missing Birth Weight (%)",
    "missingbirthwtrate": "Missing Birth Weight (%)",
    "missingbirthwt%": "Missing Birth Weight (%)",
    "missing birth wt rate": "Missing Birth Weight (%)",
    "missing birth weight %": "Missing Birth Weight (%)",
    "missing bw %": "Missing Birth Weight (%)",
    "missing birth weight rate": "Missing Birth Weight (%)",
    "missing birthweight rate": "Missing Birth Weight (%)",
    "missing bw rate": "Missing Birth Weight (%)",
    "missing weight rate": "Missing Birth Weight (%)",
    "missing birthweight %": "Missing Birth Weight (%)",
    "missing birth wt %": "Missing Birth Weight (%)",
    "missing weight %": "Missing Birth Weight (%)",
    "missing discharge status rate": "Missing Status of Discharge (%)",
    "missing status rate": "Missing Status of Discharge (%)",
    "missing discharge outcome rate": "Missing Status of Discharge (%)",
    "missing temp rate": "Missing Temperature (%)",
    "missing temperature rate": "Missing Temperature (%)",
    "missing temp %": "Missing Temperature (%)",
    "missing temperature %": "Missing Temperature (%)",
    "missing birth location rate": "Missing Birth Location (%)",
    "missing birth location %": "Missing Birth Location (%)",
    "missing discharge status %": "Missing Status of Discharge (%)",
    "missing birht weight": "Missing Birth Weight (%)",
    "missing birht weight rate": "Missing Birth Weight (%)",
    "missing birth wieght": "Missing Birth Weight (%)",
    "missing birthwieght": "Missing Birth Weight (%)",
    "missing birt weight": "Missing Birth Weight (%)",
}


NEWBORN_KPI_DEFINITIONS = {
    "Inborn Rate (%)": {
        "description": "The share of admitted newborns who were delivered in the same facility where they were admitted.",
        "numerator": "Inborn Babies",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Higher rates usually indicate that the facility is handling more of its own deliveries within the newborn care pathway.",
    },
    "Outborn Rate (%)": {
        "description": "The share of admitted newborns who were born outside the facility and then referred or brought in for care.",
        "numerator": "Outborn Babies",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Higher rates indicate a larger referral or transfer burden from outside the facility.",
    },
    "Hypothermia on Admission Rate (%)": {
        "description": "The percentage of admitted newborns who had hypothermia documented at admission.",
        "numerator": "Hypothermia Cases",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better thermal protection before and during admission.",
    },
    "Inborn Hypothermia Rate (%)": {
        "description": "The percentage of inborn babies with hypothermia at admission.",
        "numerator": "Inborn Hypothermia Cases",
        "denominator": "Total Inborn Babies",
        "interpretation": "Lower rates indicate better immediate newborn thermal care for babies born in the facility.",
    },
    "Outborn Hypothermia Rate (%)": {
        "description": "The percentage of outborn babies with hypothermia at admission.",
        "numerator": "Outborn Hypothermia Cases",
        "denominator": "Total Outborn Babies",
        "interpretation": "Lower rates indicate better stabilization and transfer conditions for referred newborns.",
    },
    "Neonatal Mortality Rate (%)": {
        "description": "The percentage of admitted newborns who died before discharge.",
        "numerator": "Dead Cases",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better newborn outcomes under inpatient care.",
    },
    "Admitted Newborns": {
        "description": "The total number of newborn admissions recorded in the newborn care program.",
        "value_name": "Admitted Newborns",
        "interpretation": "Tracks newborn inpatient workload and utilization.",
    },
    "Newborn Coverage Rate": {
        "description": "The percentage of aggregated newborn admissions (expected/aggregated counts) that are captured in the patient-level newborn admissions dataset.",
        "numerator": "Admitted Newborns",
        "denominator": "Aggregated Admissions",
        "interpretation": "Higher coverage indicates better capture of newborn admissions in the dashboard dataset.",
    },
    "Birth Weight Rate": {
        "description": "The distribution of admitted newborns across recorded birth weight categories.",
        "value_name": "Percentage of Newborns (%)",
        "interpretation": "Used to understand case mix by birth weight, especially low birth weight burden.",
    },
    "KMC Coverage by Birth Weight": {
        "description": "Coverage of Kangaroo Mother Care among newborns with valid birth weight records, grouped by birth weight category.",
        "numerator": "KMC Cases",
        "denominator": "Total Newborns with Valid Birth Weight",
        "interpretation": "Higher rates indicate stronger uptake of KMC for eligible newborns.",
    },
    "CPAP for RDS": {
        "description": "Coverage of CPAP among newborns with respiratory distress syndrome.",
        "numerator": "CPAP Cases",
        "denominator": "Total RDS Cases",
        "interpretation": "Higher rates suggest better access to respiratory support for newborns with RDS.",
    },
    "CPAP Coverage by Birth Weight": {
        "description": "Coverage of CPAP across newborn birth weight categories among newborns with valid birth weight data.",
        "numerator": "CPAP Cases",
        "denominator": "Total Newborns with Valid Birth Weight",
        "interpretation": "Used to assess whether respiratory support is reaching the newborn groups that need it most.",
    },
    "Missing Temperature (%)": {
        "description": "The percentage of newborn admissions missing a temperature value at admission.",
        "numerator": "Patients with Missing Temperature",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better admission documentation quality.",
    },
    "Missing Birth Weight (%)": {
        "description": "The percentage of newborn admissions missing birth weight documentation.",
        "numerator": "Patients with Missing Birth Weight",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better clinical documentation and stronger newborn monitoring.",
    },
    "Missing Status of Discharge (%)": {
        "description": "The percentage of newborn discharge records missing newborn status at discharge.",
        "numerator": "Patients with Missing Status",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better discharge outcome documentation.",
    },
    "Missing Discharge Status (%)": {
        "description": "The percentage of newborn discharge records missing newborn status at discharge.",
        "numerator": "Patients with Missing Status",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better discharge outcome documentation.",
    },
    "Missing Birth Location (%)": {
        "description": "The percentage of newborn admissions missing birth location documentation.",
        "numerator": "Patients with Missing Birth Location",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better capture of referral and delivery origin information.",
    },
}


NEWBORN_HELP_EXAMPLES = [
    "Plot Admitted Newborns this year",
    "Plot Newborn Coverage Rate last year",
    "Show Neonatal Mortality Rate for my facility",
    "Compare Hypothermia on Admission Rate for Amhara and Tigray",
    "Define KMC Coverage by Birth Weight",
    "List newborn indicators",
    "Plot CPAP for RDS as bar chart",
    "Compare Admitted Newborns by facility in Tigray",
    "Show Birth Weight Rate in table format",
]


def get_newborn_chatbot_config():
    return {
        "program_key": "newborn",
        "label": "Newborn",
        "kpi_mapping": NEWBORN_KPI_MAPPING,
        "kpi_options": NEWBORN_KPI_OPTIONS,
        "kpi_aliases": NEWBORN_KPI_ALIASES,
        "kpi_definitions": NEWBORN_KPI_DEFINITIONS,
        "count_indicators": {"Admitted Newborns"},
        "examples": NEWBORN_HELP_EXAMPLES,
    }


def get_newborn_welcome_message(role):
    scope_map = {
        "facility": "for your facility",
        "regional": "for the facilities in your region",
        "national": "across all facilities",
        "admin": "across the available facilities",
    }
    scope = scope_map.get(role, "for the facilities you can access")
    examples = "\n".join(f"- `{example}`" for example in NEWBORN_HELP_EXAMPLES)
    return (
        f"**Newborn program selected.**\n\n"
        f"I am now using newborn indicators and newborn dashboard logic {scope}.\n\n"
        f"Available newborn indicators include:\n"
        f"- Inborn Rate (%)\n"
        f"- Outborn Rate (%)\n"
        f"- Hypothermia on Admission Rate (%)\n"
        f"- Neonatal Mortality Rate (%)\n"
        f"- Admitted Newborns\n"
        f"- Newborn Coverage Rate\n"
        f"- Birth Weight Rate\n"
        f"- KMC Coverage by Birth Weight\n"
        f"- CPAP for RDS\n"
        f"- CPAP Coverage by Birth Weight\n"
        f"- Missing Temperature (%)\n"
        f"- Missing Birth Weight (%)\n"
        f"- Missing Status of Discharge (%)\n"
        f"- Missing Birth Location (%)\n\n"
        f"Try asking:\n{examples}\n\n"
        f"Type `maternal` any time to switch programs."
    )


def validate_newborn_indicator(indicator_name):
    """
    Validates if an indicator exists in the newborn knowledge base.
    Enforces the following rules:
    - Only responds to valid newborn indicators.
    - Checks if the indicator exists in the knowledge base.
    - If it does not exist, responds exactly with the required message.
    - Does not generate random charts or guesses for unknown/maternal indicators.
    - Never accesses .env files, CSV files, or local/system data.
    - Keeps responses clear, professional, and helpful.
    """
    if not indicator_name or indicator_name not in NEWBORN_KPI_MAPPING:
        return False, "This indicator is not recognized in newborn mode. Please switch to maternal mode or type 'list indicators' to see available options."
    return True, None
