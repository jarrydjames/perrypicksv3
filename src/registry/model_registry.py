"""
Model Registry for PerryPicks v3.

Implements model version tracking, metadata storage, and retrieval.
Stores model artifacts, hyperparameters, metrics, and lineage.

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 7
"""

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np


class ModelMetadata:
    """
    Model metadata container.

    Attributes:
        model_id: Unique model identifier (hash)
        model_name: Human-readable model name
        version: Model version string
        created_at: Timestamp of model creation
        hyperparameters: Model hyperparameters (dict)
        metrics: Model performance metrics (dict)
        dataset_info: Dataset information used for training
        features: List of feature names used
        target: Target variable name
        model_type: Type of model (e.g., 'ridge', 'cqr')
        is_baseline: Whether this is a baseline model
        is_deployed: Whether model is currently deployed
        tags: List of tags for model categorization
        notes: Free-form notes about model
        lineage: Model lineage information (parent models, etc.)
        file_path: Path to model artifact file
    """

    def __init__(
        self,
        model_name: str,
        version: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        features: List[str],
        target: str,
        model_type: str = "unknown",
        is_baseline: bool = False,
        is_deployed: bool = False,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.version = version
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.dataset_info = dataset_info
        self.features = features
        self.target = target
        self.model_type = model_type
        self.is_baseline = is_baseline
        self.is_deployed = is_deployed
        self.tags = tags or []
        self.notes = notes
        self.lineage = lineage or {}
        
        # Generate model ID (hash of key metadata)
        id_string = f"{model_name}_{version}_{datetime.now(timezone.utc).isoformat()}"
        self.model_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        
        # Timestamps
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        
        # File path (set by registry)
        self.file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "dataset_info": self.dataset_info,
            "features": self.features,
            "target": self.target,
            "model_type": self.model_type,
            "is_baseline": self.is_baseline,
            "is_deployed": self.is_deployed,
            "tags": self.tags,
            "notes": self.notes,
            "lineage": self.lineage,
            "file_path": self.file_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from dictionary."""
        metadata = cls(
            model_name=data["model_name"],
            version=data["version"],
            hyperparameters=data["hyperparameters"],
            metrics=data["metrics"],
            dataset_info=data["dataset_info"],
            features=data["features"],
            target=data["target"],
            model_type=data.get("model_type", "unknown"),
            is_baseline=data.get("is_baseline", False),
            is_deployed=data.get("is_deployed", False),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            lineage=data.get("lineage", {}),
        )
        metadata.model_id = data["model_id"]
        metadata.created_at = data["created_at"]
        metadata.updated_at = data.get("updated_at", metadata.created_at)
        metadata.file_path = data.get("file_path")
        return metadata


class ModelRegistry:
    """
    Model Registry for tracking models, metadata, and artifacts.

    Implements version tracking, metadata storage, and retrieval.
    Supports multiple model types and comparison.

    Reference: execution_specification Section 7.1, 7.2
    """

    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store model registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata index file
        self.index_file = self.registry_dir / "index.json"
        
        # Models directory
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing index
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _load_index(self):
        """Load model index from file."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
    
    def _save_index(self):
        """Save model index to file."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> str:
        """
        Register a model in the registry.
        
        Args:
            model: Model object (pickle-serializable)
            metadata: Model metadata
            overwrite: Whether to overwrite existing model
        
        Returns:
            Model ID
        """
        model_id = metadata.model_id
        
        # Check if model already exists
        if model_id in self.index and not overwrite:
            raise ValueError(f"Model {model_id} already exists. Use overwrite=True to replace.")
        
        # Save model artifact
        model_filename = f"{model_id}.pkl"
        model_path = self.models_dir / model_filename
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        metadata.file_path = str(model_path)
        
        # Update index
        self.index[model_id] = metadata.to_dict()
        self._save_index()
        
        return model_id
    
    def get_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve model and metadata from registry.
        
        Args:
            model_id: Model ID to retrieve
        
        Returns:
            Tuple of (model_object, metadata)
        """
        if model_id not in self.index:
            raise ValueError(f"Model {model_id} not found in registry.")
        
        # Load metadata
        metadata = ModelMetadata.from_dict(self.index[model_id])
        
        # Load model artifact
        model_path = Path(metadata.file_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def get_metadata(self, model_id: str) -> ModelMetadata:
        """
        Retrieve metadata for a model.
        
        Args:
            model_id: Model ID to retrieve
        
        Returns:
            Model metadata
        """
        if model_id not in self.index:
            raise ValueError(f"Model {model_id} not found in registry.")
        
        return ModelMetadata.from_dict(self.index[model_id])
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        is_baseline: Optional[bool] = None,
        is_deployed: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List models in registry with optional filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            is_baseline: Filter by baseline status
            is_deployed: Filter by deployed status
            tags: Filter by tags (must have all tags)
            limit: Maximum number of models to return
        
        Returns:
            List of model metadata dictionaries
        """
        models = list(self.index.values())
        
        # Apply filters
        if model_name is not None:
            models = [m for m in models if m["model_name"] == model_name]
        
        if model_type is not None:
            models = [m for m in models if m.get("model_type") == model_type]
        
        if is_baseline is not None:
            models = [m for m in models if m.get("is_baseline") == is_baseline]
        
        if is_deployed is not None:
            models = [m for m in models if m.get("is_deployed") == is_deployed]
        
        if tags is not None:
            models = [
                m for m in models
                if all(tag in m.get("tags", []) for tag in tags)
            ]
        
        # Sort by created_at (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply limit
        if limit is not None:
            models = models[:limit]
        
        return models
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "mae"
    ) -> pd.DataFrame:
        """
        Compare models by metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare (e.g., 'mae', 'rmse')
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for model_id in model_ids:
            metadata = self.get_metadata(model_id)
            comparison.append({
                "model_id": model_id,
                "model_name": metadata.model_name,
                "version": metadata.version,
                "model_type": metadata.model_type,
                "is_baseline": metadata.is_baseline,
                "is_deployed": metadata.is_deployed,
                metric: metadata.metrics.get(metric, np.nan),
                "created_at": metadata.created_at,
            })
        
        df = pd.DataFrame(comparison)
        
        # Sort by metric (ascending for MAE/RMSE)
        if metric in ["mae", "rmse", "mse"]:
            df = df.sort_values(by=metric, ascending=True)
        
        return df
    
    def deploy_model(self, model_id: str, undeploy_others: bool = True) -> None:
        """
        Deploy a model.
        
        Args:
            model_id: Model ID to deploy
            undeploy_others: Whether to undeploy other models
        """
        if model_id not in self.index:
            raise ValueError(f"Model {model_id} not found in registry.")
        
        if undeploy_others:
            # Undeploy all other models
            for m_id, m_data in self.index.items():
                if m_id != model_id:
                    m_data["is_deployed"] = False
        
        # Deploy target model
        self.index[model_id]["is_deployed"] = True
        self._save_index()
    
    def get_deployed_model(self) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Get currently deployed model.
        
        Returns:
            Tuple of (model_object, metadata) or None if no model deployed
        """
        deployed_models = [
            m_id for m_id, m_data in self.index.items()
            if m_data.get("is_deployed", False)
        ]
        
        if not deployed_models:
            return None
        
        if len(deployed_models) > 1:
            # Multiple deployed models, return the most recent
            deployed_models.sort(
                key=lambda m_id: self.index[m_id]["created_at"],
                reverse=True
            )
        
        return self.get_model(deployed_models[0])
    
    def delete_model(self, model_id: str, force: bool = False) -> None:
        """
        Delete a model from registry.
        
        Args:
            model_id: Model ID to delete
            force: Whether to delete deployed model
        """
        if model_id not in self.index:
            raise ValueError(f"Model {model_id} not found in registry.")
        
        metadata = self.get_metadata(model_id)
        
        # Check if deployed
        if metadata.is_deployed and not force:
            raise ValueError(
                f"Model {model_id} is deployed. Use force=True to delete."
            )
        
        # Delete model artifact
        if metadata.file_path:
            Path(metadata.file_path).unlink(missing_ok=True)
        
        # Remove from index
        del self.index[model_id]
        self._save_index()



from .model_lineage import ModelLineage, LineageGraph


class ModelRegistryExtended(ModelRegistry):
    """
    Extended Model Registry with lineage support.
    
    Combines model registry with lineage tracking
    for comprehensive model management.
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        """Initialize extended model registry."""
        super().__init__(registry_dir)
        self.lineage_graph: Optional[LineageGraph] = None
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        lineage: Optional[ModelLineage] = None,
        overwrite: bool = False
    ) -> str:
        """
        Register a model with lineage information.
        
        Args:
            model: Model object
            metadata: Model metadata
            lineage: Model lineage information (optional)
            overwrite: Whether to overwrite existing model
        
        Returns:
            Model ID
        """
        # Update metadata with lineage
        if lineage:
            metadata.lineage = lineage.to_dict()
        
        # Register model
        model_id = super().register_model(model, metadata, overwrite)
        
        # Rebuild lineage graph
        self._rebuild_lineage_graph()
        
        return model_id
    
    def _rebuild_lineage_graph(self):
        """Rebuild lineage graph."""
        self.lineage_graph = LineageGraph(self)
    
    def get_lineage_graph(self) -> LineageGraph:
        """Get lineage graph."""
        if self.lineage_graph is None:
            self._rebuild_lineage_graph()
        return self.lineage_graph
    
    def visualize_lineage(self, output_file: Optional[str] = None) -> str:
        """
        Visualize model lineage.
        
        Args:
            output_file: File to save visualization
        
        Returns:
            Graphviz DOT format string
        """
        graph = self.get_lineage_graph()
        return graph.visualize(output_file)


if __name__ == "__main__":
    # Test model registry
    print("Testing Model Registry...")
    
    # Create registry
    registry = ModelRegistry(registry_dir="test_model_registry")
    
    # Create sample metadata
    metadata = ModelMetadata(
        model_name="ridge_regression",
        version="v1.0.0",
        hyperparameters={"alpha": 2.0, "solver": "auto"},
        metrics={"mae": 9.53, "rmse": 12.34, "r2": 0.65},
        dataset_info={
            "n_samples": 10000,
            "n_features": 12,
            "dataset": "halftime_with_temporal_features_total.parquet",
            "checksum": "0b8b8bffc5916f58",
        },
        features=["h1_home", "h1_away", "h1_total", "h1_margin"],
        target="h2_total",
        model_type="ridge",
        is_baseline=True,
        is_deployed=False,
        tags=["baseline", "ridge"],
        notes="Initial baseline Ridge regression model.",
    )
    
    print(f"\nModel Metadata:")
    print(f"  Model ID: {metadata.model_id}")
    print(f"  Model Name: {metadata.model_name}")
    print(f"  Version: {metadata.version}")
    print(f"  Model Type: {metadata.model_type}")
    print(f"  Is Baseline: {metadata.is_baseline}")
    
    # Register model
    model_id = registry.register_model(
        model={"type": "ridge", "alpha": 2.0},  # Dummy model object
        metadata=metadata,
    )
    
    print(f"\nRegistered model: {model_id}")
    
    # List models
    models = registry.list_models()
    print(f"\nModels in registry: {len(models)}")
    for m in models:
        print(f"  - {m['model_name']} ({m['version']}): {m['model_id']}")
    
    # Deploy model
    registry.deploy_model(model_id)
    print(f"\nDeployed model: {model_id}")
    
    # Get deployed model
    deployed_model, deployed_metadata = registry.get_deployed_model()
    print(f"\nDeployed model:")
    print(f"  Model ID: {deployed_metadata.model_id}")
    print(f"  Model Name: {deployed_metadata.model_name}")
    print(f"  Is Deployed: {deployed_metadata.is_deployed}")
