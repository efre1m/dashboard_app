import os
import traceback
from datetime import datetime

from dotenv import load_dotenv

from add_source_column import assign_source_column
from dhis2_fetcher import CSVIntegration, DHIS2DataFetcher, DEFAULT_OUTPUT_DIR, logger


AFAR_REGION_UID = "aXRko1WzDtt"
AFAR_REGION_NAME = "Afar"
NID_PROGRAM_UID = "pLk3Ht2XMKl"
AFAR_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "afar_nid")
AFAR_OUTPUT_FILE = "regional_afara_newborn_nid.csv"


class AutomatedAfarNIDPipeline:
    """Automated pipeline in the same style as dhis2_fetcher, scoped to Afar + NID only."""

    def __init__(
        self,
        base_url: str = None,
        username: str = None,
        password: str = None,
        output_dir: str = AFAR_OUTPUT_DIR,
    ):
        env_base_url = os.getenv("DHIS2_BASE_URL")
        env_username = os.getenv("DHIS2_USERNAME")
        env_password = os.getenv("DHIS2_PASSWORD")

        if not all([base_url, username, password]) and all(
            [env_base_url, env_username, env_password]
        ):
            base_url = base_url or env_base_url.rstrip("/")
            username = username or env_username
            password = password or env_password
            logger.info("Using credentials from environment/.env")

        self.base_url = base_url
        self.username = username
        self.password = password
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.fetcher = DHIS2DataFetcher(self.base_url, self.username, self.password)

        logger.info("=" * 80)
        logger.info("AFAR NID PIPELINE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"DHIS2 URL: {self.base_url}")
        logger.info(f"Username: {self.username}")
        logger.info(f"Program UID: {NID_PROGRAM_UID}")
        logger.info(f"Region UID: {AFAR_REGION_UID} ({AFAR_REGION_NAME})")
        logger.info(f"Output Directory: {os.path.abspath(self.output_dir)}")
        logger.info("Mode: read-only DHIS2 fetch, local CSV output only")
        logger.info("=" * 80)

    def run_pipeline(self) -> bool:
        """Run Afar-only NID fetch and transformation."""
        logger.info("STARTING AFAR NID AUTOMATED PIPELINE")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("=" * 80)

        try:
            if not all([self.base_url, self.username, self.password]):
                logger.error("Missing DHIS2 credentials (DHIS2_BASE_URL, DHIS2_USERNAME, DHIS2_PASSWORD)")
                return False

            logger.info("Fetching orgUnit names...")
            orgunit_names = self.fetcher.fetch_orgunit_names()
            logger.info(f"Fetched {len(orgunit_names)} orgUnit names")

            logger.info("Fetching TEIs for Afar/NID...")
            tei_data = self.fetcher.fetch_program_data(
                NID_PROGRAM_UID,
                AFAR_REGION_UID,
                "DESCENDANTS",
                1000,
            )

            tei_count = len(tei_data.get("trackedEntityInstances", []))
            logger.info(f"Found {tei_count} TEIs")

            if tei_count == 0:
                logger.warning("No TEIs found for Afar in this program")
                return False

            events_df = CSVIntegration.create_events_dataframe(
                tei_data, NID_PROGRAM_UID, orgunit_names
            )
            logger.info(f"Created {len(events_df)} events")

            patient_df = CSVIntegration.transform_events_to_patient_level(
                events_df, NID_PROGRAM_UID
            )

            if patient_df.empty:
                logger.warning("No patient-level rows after transformation")
                return False

            # Keep compatibility with current branch methods.
            patient_df = CSVIntegration.clean_transformed_dataframe(patient_df)

            patient_df["region_uid"] = AFAR_REGION_UID
            patient_df["region_name"] = AFAR_REGION_NAME

            output_path = os.path.join(self.output_dir, AFAR_OUTPUT_FILE)
            patient_df = assign_source_column(patient_df)
            patient_df.to_csv(output_path, index=False, encoding="utf-8")

            logger.info("=" * 80)
            logger.info("AFAR NID PIPELINE COMPLETE")
            logger.info(f"Fetched TEIs from DHIS2: {tei_count}")
            logger.info(f"Output unique TEIs: {patient_df['tei_id'].nunique()}")
            logger.info(f"Output rows: {len(patient_df)}")
            logger.info(f"Output file: {output_path}")
            logger.info(f"End time: {datetime.now()}")
            logger.info("=" * 80)
            return True

        except Exception as exc:
            logger.error(f"Afar NID pipeline failed: {exc}")
            logger.error(traceback.format_exc())
            return False


def main():
    load_dotenv()
    pipeline = AutomatedAfarNIDPipeline()
    success = pipeline.run_pipeline()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
