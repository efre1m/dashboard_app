# Comprehensive KPI Definitions for Chatbot

KPI_DEFINITIONS = {
    # === MATERNAL HEALTH INDICATORS ===
    
    # Standard Clinical Indicators
    "C-Section Rate (%)": {
        "description": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus.",
        "numerator": "C-Sections",
        "denominator": "Total Deliveries"
    },
    
    "Postpartum Hemorrhage (PPH) Rate (%)": {
        "description": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth (usually >500ml for vaginal, >1000ml for C-section).",
        "numerator": "PPH Cases",
        "denominator": "Total Deliveries",
        "interpretation": "PPH is a leading cause of maternal mortality. Lower rates indicate better maternal care quality."
    },
    
    "Maternal Death Rate (per 100,000)": {
        "description": "Maternal death refers to the death of a woman while pregnant or within 42 days of termination of pregnancy, from any cause related to or aggravated by the pregnancy.",
        "numerator": "Maternal Deaths",
        "denominator": "Total Deliveries",
        "interpretation": "Calculated per 100,000 live births. Lower rates indicate better maternal health outcomes."
    },
    
    "Stillbirth Rate (%)": {
        "description": "A stillbirth is the death or loss of a baby before or during delivery.",
        "numerator": "Stillbirths",
        "denominator": "Total Newborns",
        "interpretation": "Measures fetal deaths. Lower rates indicate better antenatal and intrapartum care."
    },
    
    "Delivered women who received uterotonic (%)": {
        "description": "Uterotonics (like oxytocin) are drugs given immediately after delivery to prevent Postpartum Hemorrhage by helping the uterus contract.",
        "numerator": "Women given uterotonic",
        "denominator": "Total Deliveries",
        "interpretation": "WHO recommends uterotonic administration for all deliveries. Higher rates indicate better PPH prevention."
    },
    
    "ARV Prophylaxis Rate (%)": {
        "description": "ARV Prophylaxis tracks antiretroviral drugs given to HIV-exposed infants to prevent Mother-to-Child Transmission (PMTCT).",
        "numerator": "ARV Cases",
        "denominator": "HIV-Exposed Infants",
        "interpretation": "Higher rates indicate better PMTCT program implementation."
    },
    
    "Assisted Delivery Rate (%)": {
        "description": "Assisted delivery refers to vaginal deliveries assisted by instruments like vacuum extractors or forceps.",
        "numerator": "Assisted Deliveries",
        "denominator": "Total Deliveries",
        "interpretation": "An alternative to C-section for prolonged labor. Rates should be monitored for appropriate use."
    },
    
    "Normal Vaginal Delivery (SVD) Rate (%)": {
        "description": "SVD refers to Spontaneous Vaginal Delivery without instrumental assistance or surgery.",
        "numerator": "SVD Deliveries",
        "denominator": "Total Deliveries",
        "interpretation": "The natural mode of birth. Higher rates generally indicate healthier pregnancies and appropriate intervention use."
    },
    
    "Episiotomy Rate (%)": {
        "description": "An episiotomy is a surgical cut made in the muscle between the vagina and the anus during childbirth to assist delivery.",
        "numerator": "Episiotomy Cases",
        "denominator": "Total Vaginal Deliveries",
        "interpretation": "Should be performed only when medically necessary. WHO recommends restrictive use."
    },
    
    "Antepartum Complications Rate (%)": {
        "description": "Antepartum complications are medical conditions that arise during pregnancy before labor, such as hypertension, hemorrhage, or gestational diabetes.",
        "numerator": "Complication Cases",
        "denominator": "Total Deliveries",
        "interpretation": "Tracks pregnancy complications. Higher rates may indicate high-risk populations or better detection."
    },
    
    "Postpartum Complications Rate (%)": {
        "description": "Postpartum complications are medical conditions that arise after delivery, such as infections, hemorrhage (non-PPH specific), or other maternal distress issues.",
        "numerator": "Complication Cases",
        "denominator": "Total Deliveries",
        "interpretation": "Monitors post-delivery maternal health. Lower rates indicate better postpartum care quality."
    },
    
    # === DATA QUALITY INDICATORS ===
    
    "Missing Mode of Delivery": {
        "description": "Data Quality Metric: Tracks the percentage of delivery records where the 'Mode of Delivery' (e.g., SVD, C-Section, Assisted) was not recorded.",
        "numerator": "Deliveries with Missing MD",
        "denominator": "Total Deliveries",
        "interpretation": "Lower rates indicate better data quality. High rates suggest documentation gaps."
    },
    
    "Missing Birth Outcome": {
        "description": "Data Quality Metric: Tracks the percentage of delivery records where the 'Birth Outcome' (e.g., Live Birth, Stillbirth) was not recorded.",
        "numerator": "Missing Birth Outcomes",
        "denominator": "Total Newborns",
        "interpretation": "Lower rates indicate better data quality. Essential for accurate stillbirth tracking."
    },
    
    "Missing Condition of Discharge": {
        "description": "Data Quality Metric: Tracks the percentage of maternal discharge records where the mother's condition (e.g., Discharged Healthy, Referred, Death) was not recorded.",
        "numerator": "Missing Condition of Discharge",
        "denominator": "Total Mothers",
        "interpretation": "Lower rates indicate better data quality. Critical for tracking maternal outcomes."
    },
    
    "Missing Obstetric Condition at Delivery": {
        "description": "Data Quality Metric: Tracks the percentage of delivery records where the obstetric condition at delivery was not documented.",
        "numerator": "Missing Obstetric Condition at Delivery",
        "denominator": "Total Deliveries",
        "interpretation": "Lower rates indicate better clinical documentation. Important for tracking postpartum complications."
    },
    
    "Missing Obstetric Complications Diagnosis": {
        "description": "Data Quality Metric: Tracks the percentage of delivery records where obstetric complications diagnosis (antepartum) was not documented.",
        "numerator": "Missing Obstetric Complications Diagnosis",
        "denominator": "Total Deliveries",
        "interpretation": "Lower rates indicate better clinical documentation. Essential for tracking antepartum complications."
    },
    
    "Missing Uterotonics Given at Delivery": {
        "description": "Data Quality Metric: Tracks the percentage of delivery records where uterotonic administration was not documented.",
        "numerator": "Missing Uterotonics Given at Delivery",
        "denominator": "Total Deliveries",
        "interpretation": "Lower rates indicate better medication documentation. Critical for PPH prevention monitoring."
    },
    
    # === VOLUME INDICATORS ===
    
    "Admitted Mothers": {
        "description": "The total count of mothers admitted to the facility for delivery or pregnancy-related care.",
        "value_name": "Admitted Mothers",
        "interpretation": "Tracks facility workload and service utilization."
    },
    
    "Total Deliveries": {
        "description": "The total number of delivery events recorded, regardless of the outcome (live birth or stillbirth).",
        "value_name": "Total Deliveries",
        "interpretation": "Core denominator for most maternal health indicators."
    },
}
