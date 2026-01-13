"""
High-cardinality synthetic data generation for Featurama.

Generates millions of features for Futurama-themed entities:
- Characters (delivery crew members)
- Planets (delivery destinations)
- Deliveries (package routes and metrics)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random
from faker import Faker
import logging

logger = logging.getLogger(__name__)
fake = Faker()


class FuturamaDataGenerator:
    """Generate high-cardinality synthetic data with Futurama theme."""

    # Futurama characters
    CHARACTERS = [
        ("fry", "Philip J. Fry", "delivery_boy"),
        ("bender", "Bender Bending Rodriguez", "robot"),
        ("leela", "Turanga Leela", "captain"),
        ("professor", "Hubert J. Farnsworth", "owner"),
        ("zoidberg", "John A. Zoidberg", "doctor"),
        ("hermes", "Hermes Conrad", "bureaucrat"),
        ("amy", "Amy Wong", "intern"),
        ("scruffy", "Scruffy", "janitor"),
        ("zapp", "Zapp Brannigan", "captain"),
        ("kif", "Kif Kroker", "lieutenant"),
    ]

    # Planets in the Futurama universe
    PLANETS = [
        "Earth", "Mars", "Luna Park", "Omicron Persei 8",
        "Decapod 10", "Amphibios 9", "Chapek 9", "Amazonia",
        "The Moon", "Titan", "Neptune", "Pluto",
        "Vergon 6", "Wormulon", "Spheron 1", "Neutopia"
    ]

    # Delivery package types
    PACKAGE_TYPES = [
        "Standard", "Express", "Overnight", "Hazardous",
        "Fragile", "Perishable", "Oversized", "Quantum"
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

    def generate_entities(
        self,
        num_extra_characters: int = 1000,
        num_extra_planets: int = 100
    ) -> pd.DataFrame:
        """
        Generate entity catalog.

        Args:
            num_extra_characters: Additional random characters to generate
            num_extra_planets: Additional random planets to generate

        Returns:
            DataFrame with entity information
        """
        entities = []

        # Add main characters
        for char_id, name, role in self.CHARACTERS:
            entities.append({
                'entity_id': f"char_{char_id}_{random.randint(1000, 9999)}",
                'entity_type': 'character',
                'name': name,
                'metadata': {'role': role}
            })

        # Generate additional characters
        for i in range(num_extra_characters):
            name = fake.name()
            char_id = f"char_{fake.user_name()}_{i}"
            role = random.choice(['delivery_crew', 'customer', 'alien', 'robot'])
            entities.append({
                'entity_id': char_id,
                'entity_type': 'character',
                'name': name,
                'metadata': {'role': role}
            })

        # Add main planets
        for planet in self.PLANETS:
            planet_id = f"planet_{planet.lower().replace(' ', '_')}_{random.randint(1000, 9999)}"
            entities.append({
                'entity_id': planet_id,
                'entity_type': 'planet',
                'name': planet,
                'metadata': {'sector': f"Sector-{random.randint(1, 99)}"}
            })

        # Generate additional planets
        for i in range(num_extra_planets):
            planet_name = f"{fake.word().capitalize()} {random.randint(1, 99)}"
            planet_id = f"planet_{fake.word()}_{i}"
            entities.append({
                'entity_id': planet_id,
                'entity_type': 'planet',
                'name': planet_name,
                'metadata': {'sector': f"Sector-{random.randint(1, 99)}"}
            })

        logger.info(f"Generated {len(entities)} entities")
        return pd.DataFrame(entities)

    def generate_deliveries(
        self,
        num_deliveries: int = 10000,
        character_ids: List[str] = None,
        planet_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate delivery entities.

        Args:
            num_deliveries: Number of deliveries to generate
            character_ids: Available character IDs
            planet_ids: Available planet IDs

        Returns:
            DataFrame with delivery entities
        """
        if not character_ids or not planet_ids:
            raise ValueError("Must provide character_ids and planet_ids")

        deliveries = []

        for i in range(num_deliveries):
            delivery_id = f"delivery_{i:08d}"
            origin = random.choice(planet_ids)
            destination = random.choice(planet_ids)
            while destination == origin:
                destination = random.choice(planet_ids)

            assigned_to = random.choice(character_ids)
            package_type = random.choice(self.PACKAGE_TYPES)

            deliveries.append({
                'entity_id': delivery_id,
                'entity_type': 'delivery',
                'name': f"Delivery #{i:08d}",
                'metadata': {
                    'origin': origin,
                    'destination': destination,
                    'assigned_to': assigned_to,
                    'package_type': package_type
                }
            })

        logger.info(f"Generated {len(deliveries)} deliveries")
        return pd.DataFrame(deliveries)

    def generate_character_features(
        self,
        entity_ids: List[str],
        num_days: int = 365,
        readings_per_day: int = 24
    ) -> pd.DataFrame:
        """
        Generate time-series features for characters.

        Features include:
        - delivery_count: Number of deliveries completed
        - success_rate: Delivery success rate
        - distance_traveled: Distance traveled in light-years
        - efficiency_score: Overall efficiency metric
        - energy_level: Current energy/health level
        - customer_rating: Average customer rating

        Args:
            entity_ids: Character entity IDs
            num_days: Number of days of history
            readings_per_day: Readings per day

        Returns:
            DataFrame with character features
        """
        features = []
        feature_names = [
            'delivery_count',
            'success_rate',
            'distance_traveled',
            'efficiency_score',
            'energy_level',
            'customer_rating'
        ]

        start_date = datetime.now() - timedelta(days=num_days)

        logger.info(f"Generating character features for {len(entity_ids)} entities...")

        for entity_id in entity_ids:
            # Generate baseline trends
            delivery_base = random.randint(50, 200)
            success_base = random.uniform(0.7, 0.98)
            efficiency_base = random.uniform(0.6, 0.95)

            for day in range(num_days):
                for reading in range(readings_per_day):
                    timestamp = start_date + timedelta(
                        days=day,
                        hours=reading,
                        minutes=random.randint(0, 59)
                    )

                    # Add some realistic variation and trends
                    day_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
                    hour_factor = 1 + 0.2 * np.sin(2 * np.pi * reading / 24)  # Daily pattern
                    noise = random.gauss(1, 0.1)

                    # Generate correlated features
                    delivery_count = int(delivery_base * day_factor * noise)
                    success_rate = min(0.99, success_base + random.gauss(0, 0.05))
                    distance = random.uniform(100, 10000) * day_factor
                    efficiency = efficiency_base * success_rate * hour_factor
                    energy = random.uniform(0.4, 1.0) * hour_factor
                    rating = min(5.0, max(1.0, 4.5 * success_rate + random.gauss(0, 0.3)))

                    values = [
                        delivery_count,
                        success_rate,
                        distance,
                        efficiency,
                        energy,
                        rating
                    ]

                    for feature_name, value in zip(feature_names, values):
                        features.append({
                            'entity_id': entity_id,
                            'feature_name': feature_name,
                            'value': value,
                            'timestamp': timestamp
                        })

        logger.info(f"Generated {len(features)} character feature values")
        return pd.DataFrame(features)

    def generate_planet_features(
        self,
        entity_ids: List[str],
        num_days: int = 365,
        readings_per_day: int = 4
    ) -> pd.DataFrame:
        """
        Generate time-series features for planets.

        Features include:
        - incoming_deliveries: Number of incoming deliveries
        - outgoing_deliveries: Number of outgoing deliveries
        - traffic_level: Overall traffic level
        - weather_index: Weather condition index
        - population_density: Population density metric

        Args:
            entity_ids: Planet entity IDs
            num_days: Number of days of history
            readings_per_day: Readings per day

        Returns:
            DataFrame with planet features
        """
        features = []
        feature_names = [
            'incoming_deliveries',
            'outgoing_deliveries',
            'traffic_level',
            'weather_index',
            'population_density'
        ]

        start_date = datetime.now() - timedelta(days=num_days)

        logger.info(f"Generating planet features for {len(entity_ids)} entities...")

        for entity_id in entity_ids:
            # Generate baseline characteristics
            traffic_base = random.uniform(0.3, 1.0)
            population_base = random.uniform(1000, 1000000)

            for day in range(num_days):
                for reading in range(readings_per_day):
                    timestamp = start_date + timedelta(
                        days=day,
                        hours=reading * (24 // readings_per_day),
                        minutes=random.randint(0, 59)
                    )

                    # Seasonal and random variations
                    seasonal = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                    noise = random.gauss(1, 0.15)

                    incoming = int(random.uniform(10, 200) * traffic_base * seasonal * noise)
                    outgoing = int(random.uniform(10, 200) * traffic_base * seasonal * noise)
                    traffic = traffic_base * seasonal * noise
                    weather = random.uniform(0, 1)
                    population = population_base * (1 + 0.01 * day / 365)  # Slight growth

                    values = [incoming, outgoing, traffic, weather, population]

                    for feature_name, value in zip(feature_names, values):
                        features.append({
                            'entity_id': entity_id,
                            'feature_name': feature_name,
                            'value': value,
                            'timestamp': timestamp
                        })

        logger.info(f"Generated {len(features)} planet feature values")
        return pd.DataFrame(features)

    def generate_delivery_features(
        self,
        entity_ids: List[str],
        num_readings_per_delivery: int = 50
    ) -> pd.DataFrame:
        """
        Generate features for deliveries.

        Features include:
        - estimated_duration: Estimated delivery time (hours)
        - actual_duration: Actual delivery time (hours)
        - distance: Distance in light-years
        - fuel_consumption: Fuel consumed
        - delay_minutes: Delay in minutes (can be negative for early)
        - hazard_level: Danger level encountered
        - package_weight: Weight of package

        Args:
            entity_ids: Delivery entity IDs
            num_readings_per_delivery: Status updates per delivery

        Returns:
            DataFrame with delivery features
        """
        features = []

        logger.info(f"Generating delivery features for {len(entity_ids)} entities...")

        for entity_id in entity_ids:
            # Generate delivery characteristics
            base_distance = random.uniform(100, 50000)
            estimated_duration = base_distance / random.uniform(500, 2000)  # Speed variation
            actual_duration = estimated_duration * random.uniform(0.8, 1.5)  # Variation
            delay = (actual_duration - estimated_duration) * 60  # Convert to minutes

            package_weight = random.uniform(1, 1000)
            fuel_consumption = base_distance * package_weight * random.uniform(0.001, 0.01)
            hazard_level = random.uniform(0, 1)

            # Generate readings throughout delivery lifecycle
            start_time = datetime.now() - timedelta(days=random.randint(1, 365))

            for reading in range(num_readings_per_delivery):
                timestamp = start_time + timedelta(hours=reading * (actual_duration / num_readings_per_delivery))

                # Features that might change during delivery
                current_hazard = hazard_level * random.uniform(0.5, 1.5)
                progress = reading / num_readings_per_delivery

                feature_values = {
                    'estimated_duration': estimated_duration,
                    'actual_duration': actual_duration if reading == num_readings_per_delivery - 1 else None,
                    'distance': base_distance,
                    'fuel_consumption': fuel_consumption * progress,
                    'delay_minutes': delay if reading == num_readings_per_delivery - 1 else delay * progress,
                    'hazard_level': current_hazard,
                    'package_weight': package_weight
                }

                for feature_name, value in feature_values.items():
                    if value is not None:
                        features.append({
                            'entity_id': entity_id,
                            'feature_name': feature_name,
                            'value': value,
                            'timestamp': timestamp
                        })

        logger.info(f"Generated {len(features)} delivery feature values")
        return pd.DataFrame(features)

    def generate_all_features(
        self,
        num_characters: int = 1000,
        num_planets: int = 100,
        num_deliveries: int = 10000,
        days_of_history: int = 30,
        character_readings_per_day: int = 24,
        planet_readings_per_day: int = 4,
        delivery_readings: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with entities and features.

        Args:
            num_characters: Number of characters to generate
            num_planets: Number of planets to generate
            num_deliveries: Number of deliveries to generate
            days_of_history: Days of historical data
            character_readings_per_day: Feature readings per day for characters
            planet_readings_per_day: Feature readings per day for planets
            delivery_readings: Feature readings per delivery

        Returns:
            Tuple of (entities_df, features_df)
        """
        logger.info("Generating complete Featurama dataset...")

        # Generate entities
        entities = []

        # Characters
        char_entities = self.generate_entities(
            num_extra_characters=num_characters,
            num_extra_planets=num_planets
        )
        entities.append(char_entities)

        # Extract entity IDs by type
        character_ids = char_entities[char_entities['entity_type'] == 'character']['entity_id'].tolist()
        planet_ids = char_entities[char_entities['entity_type'] == 'planet']['entity_id'].tolist()

        # Deliveries
        delivery_entities = self.generate_deliveries(
            num_deliveries=num_deliveries,
            character_ids=character_ids[:10],  # Assign to main characters
            planet_ids=planet_ids[:16]  # Use main planets
        )
        entities.append(delivery_entities)

        entities_df = pd.concat(entities, ignore_index=True)

        # Generate features
        features = []

        # Sample entities for feature generation (to manage volume)
        sampled_characters = random.sample(character_ids, min(100, len(character_ids)))
        sampled_planets = random.sample(planet_ids, min(50, len(planet_ids)))
        sampled_deliveries = delivery_entities['entity_id'].tolist()[:1000]  # First 1000 deliveries

        char_features = self.generate_character_features(
            sampled_characters,
            num_days=days_of_history,
            readings_per_day=character_readings_per_day
        )
        features.append(char_features)

        planet_features = self.generate_planet_features(
            sampled_planets,
            num_days=days_of_history,
            readings_per_day=planet_readings_per_day
        )
        features.append(planet_features)

        delivery_features = self.generate_delivery_features(
            sampled_deliveries,
            num_readings_per_delivery=delivery_readings
        )
        features.append(delivery_features)

        features_df = pd.concat(features, ignore_index=True)

        logger.info(f"Generated {len(entities_df)} entities and {len(features_df)} features")

        return entities_df, features_df

