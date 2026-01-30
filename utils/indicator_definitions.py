# utils/indicator_definitions.py
"""
Comprehensive definitions database for all health indicators.
Used by the chatbot to provide detailed explanations of indicators.
"""

INDICATOR_DEFINITIONS = {
    # ==================== MATERNAL HEALTH INDICATORS ====================
    
    "C-Section Rate (%)": {
        "definition": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus. The C-section rate measures the percentage of deliveries performed via C-section.",
        "computation": "Number of C-section deliveries / Total deliveries × 100",
        "numerator": "Deliveries with mode of delivery code '2' (C-section)",
        "denominator": "Total deliveries in the period",
        "clinical_note": "WHO recommends C-section rates between 10-15%. Rates above 15% may indicate overuse, while rates below 10% may suggest limited access to necessary surgical interventions.",
        "data_source": "mode_of_delivery_maternal_delivery_summary field"
    },
    
    "Postpartum Hemorrhage (PPH) Rate (%)": {
        "definition": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth, typically >500ml for vaginal delivery or >1000ml for C-section. It is a leading cause of maternal mortality worldwide.",
        "computation": "Number of PPH cases / Total deliveries × 100",
        "numerator": "Women with obstetric condition code '3' (PPH) at delivery",
        "denominator": "Total deliveries in the period",
        "clinical_note": "PPH is the leading cause of maternal death globally, accounting for about 27% of maternal deaths. Early detection and management are critical.",
        "data_source": "obstetric_condition_at_delivery_delivery_summary field"
    },
    
    "Normal Vaginal Delivery (SVD) Rate (%)": {
        "definition": "Spontaneous Vaginal Delivery (SVD) is a normal, unassisted vaginal birth without surgical intervention or instruments. This rate measures the percentage of deliveries that occur naturally.",
        "computation": "Number of SVD deliveries / Total deliveries × 100",
        "numerator": "Deliveries with mode of delivery code '1' (SVD)",
        "denominator": "Total deliveries in the period",
        "clinical_note": "SVD is generally associated with lower maternal and neonatal complications compared to operative deliveries. Higher SVD rates often indicate good maternal health and appropriate use of interventions.",
        "data_source": "mode_of_delivery_maternal_delivery_summary field"
    },
    
    "Maternal Death Rate (per 100,000)": {
        "definition": "Maternal death is defined as the death of a woman while pregnant or within 42 days of termination of pregnancy, from any cause related to or aggravated by the pregnancy or its management.",
        "computation": "Number of maternal deaths / Total live births × 100,000",
        "numerator": "Women with condition of discharge code '2' (Death)",
        "denominator": "Total live births in the period",
        "clinical_note": "The global target is to reduce maternal mortality ratio to less than 70 per 100,000 live births by 2030 (SDG 3.1). This indicator follows that standard scale.",
        "data_source": "condition_of_discharge_discharge_summary field"
    },
    
    "Postpartum Complications Rate (%)": {
        "definition": "Postpartum complications are medical conditions that occur during or after delivery. These include hemorrhage, sepsis, eclampsia, and other life-threatening conditions.",
        "computation": "Number of women with postpartum complications / Total deliveries × 100",
        "numerator": "Women with any documented postpartum complication code (1-10)",
        "denominator": "Total deliveries in the period",
        "clinical_note": "Most maternal deaths occur during the postpartum period. Prompt identification and management of these complications are essential for maternal survival.",
        "data_source": "obstetric_condition_at_delivery_delivery_summary field"
    },
    
    "Stillbirth Rate (%)": {
        "definition": "A stillbirth is the death or loss of a baby before or during delivery, typically defined as fetal death at ≥28 weeks of gestation or ≥1000g birth weight.",
        "computation": "Number of stillbirths / Total births × 100",
        "numerator": "Newborns with birth outcome code '2' (Stillbirth)",
        "denominator": "Total newborns (live births + stillbirths)",
        "clinical_note": "Stillbirth is a major indicator of maternal and fetal health. Many stillbirths are preventable with quality antenatal care and skilled attendance at delivery.",
        "data_source": "birth_outcome_delivery_summary and related newborn fields"
    },
    
    "Admitted Mothers": {
        "definition": "Total count of mothers admitted to the facility for delivery or pregnancy-related care during the specified period.",
        "computation": "Count of unique patient enrollments",
        "numerator": "Unique mothers with enrollment dates in the period",
        "denominator": "N/A (this is a count, not a rate)",
        "clinical_note": "This indicator tracks facility utilization and service delivery volume. It includes all maternal admissions regardless of delivery outcome.",
        "data_source": "enrollment_date field with unique tei_id"
    },
    
    "Episiotomy Rate (%)": {
        "definition": "Episiotomy is a surgical incision made in the perineum during childbirth to enlarge the vaginal opening. This rate measures how frequently this procedure is performed during vaginal deliveries.",
        "computation": "Number of episiotomies / Total vaginal deliveries × 100",
        "numerator": "Vaginal deliveries with episiotomy performed",
        "denominator": "Total vaginal deliveries (excludes C-sections)",
        "clinical_note": "WHO recommends selective (not routine) use of episiotomy. Rates above 10% may indicate overuse. Routine episiotomy is not recommended as it doesn't prevent severe perineal tears.",
        "data_source": "episiotomy-related fields in delivery summary"
    },
    
    "Antepartum Complications Rate (%)": {
        "definition": "Antepartum complications are medical conditions or complications that occur during pregnancy before labor begins. These may include conditions like pre-eclampsia, gestational diabetes, placental problems, etc.",
        "computation": "Number of deliveries with antepartum complications / Total deliveries × 100",
        "numerator": "Women with documented antepartum complications",
        "denominator": "Total deliveries in the period",
        "clinical_note": "Early detection and management of antepartum complications through quality antenatal care can significantly reduce maternal and neonatal morbidity and mortality.",
        "data_source": "Antepartum complications fields in maternal records"
    },
    
    "ARV Prophylaxis Rate (%)": {
        "definition": "Antiretroviral (ARV) prophylaxis for HIV-exposed infants is medication given to newborns born to HIV-positive mothers to prevent mother-to-child transmission of HIV.",
        "computation": "Number of HIV-exposed infants receiving ARV / Total HIV-exposed infants × 100",
        "numerator": "HIV-exposed newborns who received ARV prophylaxis",
        "denominator": "Total newborns born to HIV-positive mothers",
        "clinical_note": "ARV prophylaxis can reduce mother-to-child HIV transmission to less than 1% when combined with other PMTCT interventions. This is a critical indicator for PMTCT program effectiveness.",
        "data_source": "hiv_result_delivery_summary and arv_rx_for_newborn fields"
    },
    
    "Assisted Delivery Rate (%)": {
        "definition": "Assisted delivery (also called instrumental delivery) uses instruments like forceps or vacuum extractors to help deliver the baby vaginally when complications arise during labor.",
        "computation": "Number of assisted deliveries / Total deliveries × 100",
        "numerator": "Deliveries using forceps, vacuum, or other instruments",
        "denominator": "Total deliveries in the period",
        "clinical_note": "Assisted delivery is used when vaginal delivery is possible but requires assistance, avoiding the need for C-section. Proper training is essential to minimize complications.",
        "data_source": "instrumental_delivery_form field"
    },
    
    "Delivered women who received uterotonic (%)": {
        "definition": "Uterotonics (like oxytocin) are medications given immediately after delivery to help the uterus contract and prevent postpartum hemorrhage. This is part of Active Management of Third Stage of Labor (AMTSL).",
        "computation": "Number of women receiving uterotonics / Total deliveries × 100",
        "numerator": "Women who received uterotonic medication at delivery",
        "denominator": "Total deliveries in the period",
        "clinical_note": "WHO strongly recommends uterotonic administration for all births to prevent PPH. Oxytocin is the preferred uterotonic. This is one of the most effective interventions to reduce maternal mortality.",
        "data_source": "uterotonics_given_delivery_summary field"
    },
    
    "Early Postnatal Care (PNC) Coverage (%)": {
        "definition": "Early Postnatal Care refers to medical care provided to the mother and newborn within the first 48 hours after delivery. This critical period is when most maternal and neonatal deaths occur.",
        "computation": "Number of mothers receiving early PNC / Total deliveries × 100",
        "numerator": "Mothers who received PNC within 48 hours of delivery",
        "denominator": "Total deliveries in the period",
        "clinical_note": "The first 48 hours after birth are the most critical for detecting and managing complications. WHO recommends at least 3 postnatal contacts in the first 6 weeks.",
        "data_source": "date_stay_pp_postpartum_care field"
    },
    
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
        "definition": "IPPCAR measures the percentage of women who accept and receive a family planning method before discharge after delivery. This helps prevent closely spaced pregnancies.",
        "computation": "Number of women accepting postpartum FP / Total deliveries × 100",
        "numerator": "Women who accepted family planning method postpartum",
        "denominator": "Total deliveries in the period",
        "clinical_note": "Postpartum family planning is crucial for birth spacing. Short intervals between pregnancies (<18 months) increase risks for both mother and baby.",
        "data_source": "fp_counseling_and_method_provided_pp_postpartum_care field"
    },
    
    # ==================== DATA QUALITY INDICATORS ====================
    
    "Missing Mode of Delivery": {
        "definition": "Percentage of delivery records where the mode of delivery (how the baby was delivered) was not documented. This is a data quality indicator.",
        "computation": "Number of deliveries with missing mode / Total deliveries × 100",
        "numerator": "Delivery records with no mode of delivery documented",
        "denominator": "Total deliveries in the period",
        "clinical_note": "Complete documentation is essential for quality monitoring and clinical decision-making. High rates of missing data indicate gaps in record-keeping.",
        "data_source": "mode_of_delivery_maternal_delivery_summary field"
    },
    
    "Missing Birth Outcome": {
        "definition": "Percentage of newborn records where the birth outcome (live birth or stillbirth) was not documented. This is a data quality indicator.",
        "computation": "Number of newborns with missing outcome / Total newborns × 100",
        "numerator": "Newborn records with no birth outcome documented",
        "denominator": "Total newborn records in the period",
        "clinical_note": "Birth outcome is a critical data point for calculating stillbirth rates and neonatal mortality. Missing data compromises the accuracy of these vital statistics.",
        "data_source": "birth_outcome_delivery_summary and related newborn fields"
    },
    
    "Missing Condition of Discharge": {
        "definition": "Percentage of maternal records where the condition at discharge (alive, dead, referred, etc.) was not documented. This is a data quality indicator.",
        "computation": "Number of mothers with missing discharge status / Total mothers × 100",
        "numerator": "Maternal records with no discharge condition documented",
        "denominator": "Total maternal admissions in the period",
        "clinical_note": "Discharge status is essential for tracking maternal outcomes and mortality. Missing data makes it impossible to accurately calculate maternal death rates.",
        "data_source": "condition_of_discharge_discharge_summary field"
    },
    
    # ==================== NEWBORN HEALTH INDICATORS ====================
    
    "Neonatal Mortality Rate (%)": {
        "definition": "Neonatal Mortality Rate (NMR) is the percentage of live-born babies who die within the first 28 days of life. It's a key indicator of newborn health and quality of neonatal care.",
        "computation": "Number of neonatal deaths / Total live births × 100",
        "numerator": "Newborns who died within 28 days of birth",
        "denominator": "Total live births in the period",
        "clinical_note": "Global target is to reduce NMR to 12 per 1,000 live births by 2030 (SDG 3.2). Most neonatal deaths are preventable with quality essential newborn care.",
        "data_source": "Newborn discharge status and date of death fields"
    },
    
    "Admitted Newborns": {
        "definition": "Total count of newborns admitted to the neonatal care unit during the specified period, including both inborn (born in facility) and outborn (born elsewhere) babies.",
        "computation": "Count of unique newborn admissions",
        "numerator": "Unique newborns admitted in the period",
        "denominator": "N/A (this is a count, not a rate)",
        "clinical_note": "This tracks neonatal unit utilization and service delivery volume. It includes all admissions regardless of birth location or outcome.",
        "data_source": "Newborn admission records with unique patient IDs"
    },
    
    "Inborn Rate (%)": {
        "definition": "Percentage of admitted newborns who were born within the same facility (as opposed to being transferred from another location).",
        "computation": "Number of inborn newborns / Total admitted newborns × 100",
        "numerator": "Newborns born in the facility",
        "denominator": "Total newborn admissions",
        "clinical_note": "Higher inborn rates generally indicate better access to facility-based delivery services. Inborn babies typically have better outcomes than outborn due to immediate access to care.",
        "data_source": "Birth location or admission source field"
    },
    
    "Outborn Rate (%)": {
        "definition": "Percentage of admitted newborns who were born outside the facility and then transferred in for specialized neonatal care.",
        "computation": "Number of outborn newborns / Total admitted newborns × 100",
        "numerator": "Newborns born outside and transferred to facility",
        "denominator": "Total newborn admissions",
        "clinical_note": "High outborn rates may indicate limited delivery services in the community or referral patterns. Outborn babies often arrive in more critical condition.",
        "data_source": "Birth location or admission source field"
    },
    
    "Hypothermia on Admission Rate (%)": {
        "definition": "Percentage of newborns who have a body temperature below 36.5°C when first admitted. Hypothermia is a major risk factor for neonatal mortality.",
        "computation": "Number of hypothermic newborns / Total admitted newborns × 100",
        "numerator": "Newborns with admission temperature <36.5°C",
        "denominator": "Total newborn admissions with temperature recorded",
        "clinical_note": "Hypothermia increases mortality risk 4-fold. Prevention through immediate drying, skin-to-skin contact, and warm environment is critical.",
        "data_source": "Admission temperature field"
    },
    
    "General CPAP Coverage": {
        "definition": "Continuous Positive Airway Pressure (CPAP) is a breathing support method for newborns with respiratory distress. This measures the percentage of eligible newborns receiving CPAP.",
        "computation": "Number of newborns receiving CPAP / Total eligible newborns × 100",
        "numerator": "Newborns who received CPAP therapy",
        "denominator": "Newborns with respiratory distress or other CPAP indications",
        "clinical_note": "CPAP is a life-saving intervention for respiratory distress syndrome and other breathing problems. It can reduce the need for mechanical ventilation.",
        "data_source": "CPAP treatment records"
    },
    
    "CPAP for RDS": {
        "definition": "Percentage of newborns with Respiratory Distress Syndrome (RDS) who receive CPAP therapy. RDS is common in premature babies due to immature lungs.",
        "computation": "Number of RDS cases receiving CPAP / Total RDS cases × 100",
        "numerator": "RDS patients who received CPAP",
        "denominator": "Total newborns diagnosed with RDS",
        "clinical_note": "Early CPAP for RDS significantly reduces mortality and the need for mechanical ventilation, especially in preterm infants.",
        "data_source": "RDS diagnosis and CPAP treatment records"
    },
    
    "KMC Coverage by Birth Weight": {
        "definition": "Kangaroo Mother Care (KMC) is skin-to-skin contact between mother and baby, especially for low birth weight infants. This tracks KMC provision across different birth weight categories.",
        "computation": "Number receiving KMC / Total eligible by weight category × 100",
        "numerator": "Low birth weight babies receiving KMC",
        "denominator": "Total low birth weight babies eligible for KMC",
        "clinical_note": "KMC reduces mortality in low birth weight babies by 40%. It also promotes breastfeeding, bonding, and temperature regulation.",
        "data_source": "KMC records and birth weight measurements"
    },
    
    "Birth Weight Rate": {
        "definition": "Distribution of newborns across different birth weight categories (<1500g, 1500-2499g, ≥2500g). Birth weight is a key predictor of neonatal survival and health.",
        "computation": "Percentage in each weight category",
        "numerator": "Newborns in each weight category",
        "denominator": "Total newborns with recorded birth weight",
        "clinical_note": "Low birth weight (<2500g) is associated with higher mortality and morbidity. Very low birth weight (<1500g) requires intensive care.",
        "data_source": "Birth weight measurement field"
    },
    
    "Missing Temperature (%)": {
        "definition": "Percentage of newborn admission records where temperature was not documented. This is a data quality indicator for essential newborn care.",
        "computation": "Number with missing temperature / Total admissions × 100",
        "numerator": "Newborn records with no temperature recorded",
        "denominator": "Total newborn admissions",
        "clinical_note": "Temperature is a vital sign that must be recorded at admission. Missing data indicates gaps in essential newborn care protocols.",
        "data_source": "Admission temperature field"
    },
    
    "Missing Birth Weight (%)": {
        "definition": "Percentage of newborn records where birth weight was not documented. This is a critical data quality indicator.",
        "computation": "Number with missing birth weight / Total newborns × 100",
        "numerator": "Newborn records with no birth weight recorded",
        "denominator": "Total newborn records",
        "clinical_note": "Birth weight is one of the most important predictors of neonatal outcomes. Missing data compromises clinical care and monitoring.",
        "data_source": "Birth weight field"
    },
    
    "Missing Discharge Status (%)": {
        "definition": "Percentage of newborn records where the discharge status (alive, dead, transferred, etc.) was not documented. This is a data quality indicator.",
        "computation": "Number with missing discharge status / Total newborns × 100",
        "numerator": "Newborn records with no discharge status documented",
        "denominator": "Total newborn admissions",
        "clinical_note": "Discharge status is essential for calculating neonatal mortality rates and tracking outcomes. Missing data makes it impossible to accurately assess care quality.",
        "data_source": "Discharge status field"
    },
}


def get_indicator_definition(indicator_name):
    """
    Get the full definition for an indicator.
    
    Args:
        indicator_name: Name of the indicator (exact match or fuzzy)
        
    Returns:
        Dictionary with definition details or None if not found
    """
    # Try exact match first
    if indicator_name in INDICATOR_DEFINITIONS:
        return INDICATOR_DEFINITIONS[indicator_name]
    
    # Try fuzzy match (case-insensitive)
    indicator_lower = indicator_name.lower()
    for key in INDICATOR_DEFINITIONS.keys():
        if key.lower() == indicator_lower:
            return INDICATOR_DEFINITIONS[key]
    
    return None


def format_definition_response(indicator_name, definition_dict):
    """
    Format a definition dictionary into a user-friendly response.
    
    Args:
        indicator_name: Name of the indicator
        definition_dict: Dictionary with definition details
        
    Returns:
        Formatted markdown string
    """
    if not definition_dict:
        return None
    
    response = f"## 📊 {indicator_name}\n\n"
    response += f"**Definition:**\n{definition_dict['definition']}\n\n"
    
    if definition_dict.get('computation'):
        response += f"**How it's computed:**\n{definition_dict['computation']}\n\n"
    
    if definition_dict.get('numerator') and definition_dict.get('denominator'):
        response += f"**Formula Details:**\n"
        response += f"- **Numerator:** {definition_dict['numerator']}\n"
        response += f"- **Denominator:** {definition_dict['denominator']}\n\n"
    
    if definition_dict.get('clinical_note'):
        response += f"**Clinical Significance:**\n{definition_dict['clinical_note']}\n\n"
    
    if definition_dict.get('data_source'):
        response += f"*Data source: {definition_dict['data_source']}*"
    
    return response
