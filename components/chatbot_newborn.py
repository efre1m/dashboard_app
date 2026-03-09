from newborns_dashboard.dash_co_newborn import NEWBORN_KPI_MAPPING, NEWBORN_KPI_OPTIONS


NEWBORN_KPI_ALIASES = {
    "inborn": "Inborn Rate (%)",
    "outborn": "Outborn Rate (%)",
    "neonatal death": "Neonatal Mortality Rate (%)",
    "neonatal mortality": "Neonatal Mortality Rate (%)",
    "mortality": "Neonatal Mortality Rate (%)",
    "nmr": "Neonatal Mortality Rate (%)",
    "admitted newborns": "Admitted Newborns",
    "newborn admissions": "Admitted Newborns",
    "newborn admission": "Admitted Newborns",
    "admitted babies": "Admitted Newborns",
    "admitted neonates": "Admitted Newborns",
    "kmc": "KMC Coverage by Birth Weight",
    "kangaroo": "KMC Coverage by Birth Weight",
    "kangaro": "KMC Coverage by Birth Weight",
    "kangaroo mother care": "KMC Coverage by Birth Weight",
    "skin to skin": "KMC Coverage by Birth Weight",
    "cpap": "CPAP for RDS",
    "c pap": "CPAP for RDS",
    "c-pap": "CPAP for RDS",
    "rds": "CPAP for RDS",
    "respiratory distress": "CPAP for RDS",
    "birth weight": "Birth Weight Rate",
    "birthweight": "Birth Weight Rate",
    "birht weight": "Birth Weight Rate",
    "bw": "Birth Weight Rate",
    "hypothermia": "Hypothermia on Admission Rate (%)",
    "hypthermia": "Hypothermia on Admission Rate (%)",
    "hipothermia": "Hypothermia on Admission Rate (%)",
    "hypo thermia": "Hypothermia on Admission Rate (%)",
    "temperature": "Hypothermia on Admission Rate (%)",
    "missing temperature": "Missing Temperature (%)",
    "missing temp": "Missing Temperature (%)",
    "missing birth weight": "Missing Birth Weight (%)",
    "missing weight": "Missing Birth Weight (%)",
    "missing discharge status": "Missing Status of Discharge (%)",
    "missing status of discharge": "Missing Status of Discharge (%)",
    "missing newborn status": "Missing Status of Discharge (%)",
    "missing birth location": "Missing Birth Location (%)",
    "missing delivery location": "Missing Birth Location (%)",
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
    "Missing Birth Location (%)": {
        "description": "The percentage of newborn admissions missing birth location documentation.",
        "numerator": "Patients with Missing Birth Location",
        "denominator": "Total Admitted Newborns",
        "interpretation": "Lower rates indicate better capture of referral and delivery origin information.",
    },
}


NEWBORN_HELP_EXAMPLES = [
    "Plot Admitted Newborns this year",
    "Show Neonatal Mortality Rate for my facility",
    "Compare Hypothermia on Admission Rate for Amhara and Tigray",
    "Define KMC Coverage by Birth Weight",
    "List newborn indicators",
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
