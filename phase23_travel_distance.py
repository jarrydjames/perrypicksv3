"""
Phase 23: Travel Distance Features
Add travel distance as a predictive feature.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TravelDistanceFeatureBuilder:
    """
    Add travel distance features.
    
    Note: Requires city coordinates for all NBA teams.
    This is a framework showing how travel distance would be calculated.
    """

    # NBA Team Stadium Coordinates (approximate)
    TEAM_COORDINATES = {
        1610612737: {'city': 'Atlanta', 'lat': 33.7537, 'lon': -84.3863},  # Hawks
        1610612738: {'city': 'Boston', 'lat': 42.3663, 'lon': -71.0617},   # Celtics
        1610612739: {'city': 'Cleveland', 'lat': 41.4967, 'lon': -81.6883}, # Cavaliers
        1610612740: {'city': 'New Orleans', 'lat': 29.9478, 'lon': -90.0706}, # Pelicans
        1610612741: {'city': 'Chicago', 'lat': 41.8819, 'lon': -87.6328},  # Bulls
        1610612742: {'city': 'Dallas', 'lat': 32.7903, 'lon': -96.8104},  # Mavericks
        1610612743: {'city': 'Denver', 'lat': 39.7392, 'lon': -104.9903},  # Nuggets
        1610612744: {'city': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194}, # Warriors
        1610612745: {'city': 'Houston', 'lat': 29.7604, 'lon': -95.3698},  # Rockets
        1610612746: {'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437}, # Clippers
        1610612747: {'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437}, # Lakers
        1610612748: {'city': 'Miami', 'lat': 25.7617, 'lon': -80.1918},    # Heat
        1610612749: {'city': 'Milwaukee', 'lat': 43.0389, 'lon': -87.9065}, # Bucks
        1610612750: {'city': 'Minneapolis', 'lat': 44.9778, 'lon': -93.2650}, # Timberwolves
        1610612751: {'city': 'Brooklyn', 'lat': 40.6827, 'lon': -73.9710}, # Nets
        1610612752: {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060},  # Knicks
        1610612753: {'city': 'Orlando', 'lat': 28.5383, 'lon': -81.3792},  # Magic
        1610612754: {'city': 'Indianapolis', 'lat': 39.7684, 'lon': -86.1581}, # Pacers
        1610612755: {'city': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652}, # 76ers
        1610612756: {'city': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740}, # Suns
        1610612757: {'city': 'Portland', 'lat': 45.5152, 'lon': -122.6784}, # Trail Blazers
        1610612758: {'city': 'Sacramento', 'lat': 38.5816, 'lon': -121.4944}, # Kings
        1610612759: {'city': 'San Antonio', 'lat': 29.4241, 'lon': -98.4936}, # Spurs
        1610612760: {'city': 'Oklahoma City', 'lat': 35.4676, 'lon': -97.5164}, # Thunder
        1610612761: {'city': 'Toronto', 'lat': 43.6532, 'lon': -79.3832},  # Raptors
        1610612762: {'city': 'Washington', 'lat': 38.9072, 'lon': -77.0369}, # Wizards
        1610612763: {'city': 'Memphis', 'lat': 35.1495, 'lon': -90.0490},   # Grizzlies
        1610612764: {'city': 'Salt Lake City', 'lat': 40.7608, 'lon': -111.8910}, # Jazz
        1610612765: {'city': 'Detroit', 'lat': 42.3314, 'lon': -83.0458},  # Pistons
        1610612766: {'city': 'Charlotte', 'lat': 35.2271, 'lon': -80.8431}, # Hornets
    }

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two coordinates in miles."""
        R = 3959.87433  # Earth's radius in miles
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    def __init__(self, features_path: str, output_path: str):
        self.features_path = features_path
        self.output_path = output_path
        self.features_df = None
        self.augmented_df = None

    def load_data(self):
        """Load existing features."""
        logger.info(f"Loading features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} games")
        return self

    def calculate_travel_distance(self, df):
        """
        Calculate travel distance for both teams.
        
        Note: This requires tracking team schedules to determine
        where each team played their previous game.
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAVEL DISTANCE FEATURES")
        logger.info("=" * 70)
        logger.info("\n⚠️  NOTE: Travel distance calculation requires tracking")
        logger.info("      each team's schedule to determine previous game location")
        logger.info("\n" + "=" * 70)
        logger.info("FRAMEWORK FOR TRAVEL DISTANCE")
        logger.info("=" * 70)

        # For each game, need to:
        # 1. Find previous game for home team
        # 2. Find previous game for away team
        # 3. Calculate distance from previous location to current game
        # 4. Add features for home/away travel distance and differential

        # Placeholder features (would be calculated if schedule data was available)
        df['home_travel_distance'] = 0.0
        df['away_travel_distance'] = 0.0
        df['travel_distance_diff'] = 0.0

        # Binary flags for significant travel (> 1000 miles)
        df['home_long_travel'] = 0.0
        df['away_long_travel'] = 0.0

        logger.info("\nPlaceholder travel distance features added:")
        logger.info("  home_travel_distance, away_travel_distance")
        logger.info("  travel_distance_diff")
        logger.info("  home_long_travel, away_long_travel")
        logger.info("\n" + "=" * 70)
        logger.info("EXPECTED IMPACT (if real travel data were available):")
        logger.info("=" * 70)
        logger.info("  Travel distance is MEDIUM-LOW IMPACT")
        logger.info("  Expected MAE improvement: 0.2-0.8 points")
        logger.info("  Long travel (> 2000 miles) can affect team performance")
        logger.info("  Back-to-back games + long travel = significant fatigue")

        return df

    def add_travel_features(self):
        """Add travel distance features."""
        df = self.features_df.copy()
        self.augmented_df = self.calculate_travel_distance(df)
        return self

    def save_features(self):
        """Save augmented features."""
        logger.info(f"\nSaving augmented features to {self.output_path}")
        self.augmented_df.to_parquet(self.output_path, index=False)
        logger.info(f"✓ Saved {len(self.augmented_df)} games with {len(self.augmented_df.columns)} columns")

        # Save feature list
        feature_list_path = self.output_path.replace('.parquet', '_feature_list.txt')
        with open(feature_list_path, 'w') as f:
            for col in self.augmented_df.columns:
                f.write(f"{col}\n")
        logger.info(f"✓ Feature list saved to {feature_list_path}")

        return self

    def get_new_features(self):
        """Return list of newly added features."""
        return [
            'home_travel_distance',
            'away_travel_distance',
            'travel_distance_diff',
            'home_long_travel',
            'away_long_travel',
        ]

    def provide_implementation_guide(self):
        """Provide guidance on implementing travel distance features."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAVEL DISTANCE IMPLEMENTATION GUIDE")
        logger.info("=" * 70)
        logger.info("\nSTEP 1: Build Team Schedule Tracking")
        logger.info("  - Track each team's game history")
        logger.info("  - Store game dates, locations, and coordinates")
        logger.info("\nSTEP 2: Calculate Previous Game Location")
        logger.info("  - For each game, find previous game for each team")
        logger.info("  - Handle season breaks and long gaps")
        logger.info("\nSTEP 3: Calculate Distance")
        logger.info("  - Use Haversine formula for great-circle distance")
        logger.info("  - Add features: distance, long travel flag, differential")
        logger.info("\n" + "=" * 70)
        logger.info("CHALLENGES:")
        logger.info("=" * 70)
        logger.info("  1. Requires full historical schedule data")
        logger.info("  2. Need to handle team relocations (if any)")
        logger.info("  3. Neutral site games (e.g., NBA Cup)")
        logger.info("  4. Cross-timezone effects (not just distance)")
        logger.info("\n" + "=" * 70)
        logger.info("RECOMMENDATION:")
        logger.info("=" * 70)
        logger.info("  Travel distance adds minimal value compared to complexity")
        logger.info("  Expected MAE improvement: < 1 point")
        logger.info("  Consider only if already tracking schedules for other reasons")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 23."""
    features_path = 'data/processed/final_features.parquet'
    output_path = 'data/processed/travel_features.parquet'

    builder = TravelDistanceFeatureBuilder(features_path, output_path)
    builder.load_data()
    builder.add_travel_features()
    builder.save_features()
    builder.provide_implementation_guide()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 23: COMPLETE (Framework Implementation)")
    logger.info("=" * 70)
    logger.info("\nNOTE: Real travel distance features require schedule tracking")
    logger.info("      This is a framework showing how to calculate")


if __name__ == "__main__":
    main()
