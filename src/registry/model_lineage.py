"""
Model Lineage and Provenance Tracking for PerryPicks v3.

Implements model lineage tracking, provenance, and dependency graph.
Tracks parent models, training datasets, and transformation history.

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 7.3
"""

from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import json
from pathlib import Path

from .model_registry import ModelRegistry, ModelMetadata



class ModelLineage:
    """
    Model lineage and provenance tracker.

    Tracks:
    - Parent models (hyperparameter search, ensembling)
    - Training datasets and checksums
    - Transformations and feature engineering
    - Deployment history
    - Comparison with other models
    """

    def __init__(
        self,
        parent_model_ids: Optional[List[str]] = None,
        dataset_checksum: Optional[str] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        comparison_model_ids: Optional[List[str]] = None,
        deployment_history: Optional[List[Dict[str, Any]]] = None,
    ):
        self.parent_model_ids = parent_model_ids or []
        self.dataset_checksum = dataset_checksum
        self.transformations = transformations or []
        self.training_config = training_config or {}
        self.comparison_model_ids = comparison_model_ids or []
        self.deployment_history = deployment_history or []
    
    def add_parent(self, model_id: str, relationship: str = "derived"):
        """
        Add a parent model to lineage.
        
        Args:
            model_id: Parent model ID
            relationship: Type of relationship ('derived', 'ensembled', 'tuned')
        """
        self.parent_model_ids.append({
            "model_id": model_id,
            "relationship": relationship,
        })
    
    def add_transformation(
        self,
        name: str,
        parameters: Dict[str, Any],
        timestamp: Optional[str] = None
    ):
        """
        Add a transformation to lineage.
        
        Args:
            name: Name of transformation
            parameters: Transformation parameters
            timestamp: Timestamp of transformation (optional)
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        
        self.transformations.append({
            "name": name,
            "parameters": parameters,
            "timestamp": timestamp,
        })
    
    def add_deployment(
        self,
        environment: str,
        timestamp: Optional[str] = None,
        deployed_by: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """
        Add a deployment to history.
        
        Args:
            environment: Deployment environment (e.g., 'production', 'staging')
            timestamp: Timestamp of deployment (optional)
            deployed_by: User or system that deployed
            notes: Deployment notes
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        
        self.deployment_history.append({
            "environment": environment,
            "timestamp": timestamp,
            "deployed_by": deployed_by,
            "notes": notes,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parent_model_ids": self.parent_model_ids,
            "dataset_checksum": self.dataset_checksum,
            "transformations": self.transformations,
            "training_config": self.training_config,
            "comparison_model_ids": self.comparison_model_ids,
            "deployment_history": self.deployment_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelLineage":
        """Create ModelLineage from dictionary."""
        return cls(
            parent_model_ids=data.get("parent_model_ids", []),
            dataset_checksum=data.get("dataset_checksum"),
            transformations=data.get("transformations", []),
            training_config=data.get("training_config", {}),
            comparison_model_ids=data.get("comparison_model_ids", []),
            deployment_history=data.get("deployment_history", []),
        )


class LineageGraph:
    """
    Model lineage graph for visualization and analysis.
    
    Builds a dependency graph of models, showing:
    - Parent-child relationships
    - Training datasets
    - Transformations
    - Deployment history
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize lineage graph.
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
        self.graph: Dict[str, Dict[str, Any]] = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build lineage graph from registry."""
        for model_id in self.registry.index.keys():
            metadata = self.registry.get_metadata(model_id)
            lineage = ModelLineage.from_dict(metadata.lineage)
            
            self.graph[model_id] = {
                "metadata": metadata,
                "lineage": lineage,
                "parents": lineage.parent_model_ids,
                "children": [],
            }
        
        # Build children relationships
        for model_id, node in self.graph.items():
            for parent in node["parents"]:
                if isinstance(parent, dict):
                    parent_id = parent["model_id"]
                else:
                    parent_id = parent
                
                if parent_id in self.graph:
                    self.graph[parent_id]["children"].append(model_id)
    
    def get_ancestors(self, model_id: str) -> List[str]:
        """
        Get all ancestor models (direct and indirect parents).
        
        Args:
            model_id: Starting model ID
        
        Returns:
            List of ancestor model IDs
        """
        ancestors = []
        visited: Set[str] = set()
        
        def dfs(current_id: str):
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            if current_id not in self.graph:
                return
            
            for parent in self.graph[current_id]["parents"]:
                if isinstance(parent, dict):
                    parent_id = parent["model_id"]
                else:
                    parent_id = parent
                
                ancestors.append(parent_id)
                dfs(parent_id)
        
        dfs(model_id)
        return ancestors
    
    def get_descendants(self, model_id: str) -> List[str]:
        """
        Get all descendant models (direct and indirect children).
        
        Args:
            model_id: Starting model ID
        
        Returns:
            List of descendant model IDs
        """
        descendants = []
        visited: Set[str] = set()
        
        def dfs(current_id: str):
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            if current_id not in self.graph:
                return
            
            for child_id in self.graph[current_id]["children"]:
                descendants.append(child_id)
                dfs(child_id)
        
        dfs(model_id)
        return descendants
    
    def get_lineage_depth(self, model_id: str) -> int:
        """
        Get the depth of model lineage (number of ancestors).
        
        Args:
            model_id: Model ID
        
        Returns:
            Lineage depth (number of ancestor levels)
        """
        depth = 0
        visited: Set[str] = set()
        
        def dfs(current_id: str, current_depth: int):
            nonlocal depth
            depth = max(depth, current_depth)
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            if current_id not in self.graph:
                return
            
            for parent in self.graph[current_id]["parents"]:
                if isinstance(parent, dict):
                    parent_id = parent["model_id"]
                else:
                    parent_id = parent
                
                dfs(parent_id, current_depth + 1)
        
        dfs(model_id, 0)
        return depth
    
    def visualize(self, output_file: Optional[str] = None) -> str:
        """
        Generate visualization of lineage graph.
        
        Args:
            output_file: File to save visualization (optional)
        
        Returns:
            Graphviz DOT format string
        """
        lines = [
            "digraph ModelLineage {",
            "  rankdir=TD;",
            "  node [shape=box, style=rounded];",
            "",
        ]
        
        # Add nodes
        for model_id, node in self.graph.items():
            metadata = node["metadata"]
            label = f"{metadata.model_name}\n{metadata.version}"
            
            if metadata.is_deployed:
                lines.append(f"  \"{model_id}\" [label=\"{label}\", color=blue, penwidth=2.0];")
            elif metadata.is_baseline:
                lines.append(f"  \"{model_id}\" [label=\"{label}\", color=gray, style=dashed];")
            else:
                lines.append(f"  \"{model_id}\" [label=\"{label}\"];")
        
        lines.append("")
        
        # Add edges (parent -> child)
        for model_id, node in self.graph.items():
            for parent in node["parents"]:
                if isinstance(parent, dict):
                    parent_id = parent["model_id"]
                    relationship = parent.get("relationship", "derived")
                else:
                    parent_id = parent
                    relationship = "derived"
                
                if parent_id in self.graph:
                    lines.append(f"  \"{parent_id}\" -> \"{model_id}\" [label=\"{relationship}\"];")
        
        lines.append("}")
        
        dot_string = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(dot_string)
        
        return dot_string


if __name__ == "__main__":
    # Test model lineage
    print("Testing Model Lineage...")
    
    # Create lineage
    lineage = ModelLineage(
        dataset_checksum="0b8b8bffc5916f58",
        training_config={"epochs": 100, "batch_size": 32},
    )
    
    # Add transformations
    lineage.add_transformation(
        name="standard_scaler",
        parameters={"with_mean": True, "with_std": True}
    )
    lineage.add_transformation(
        name="pca",
        parameters={"n_components": 10}
    )
    
    # Add deployment
    lineage.add_deployment(
        environment="production",
        deployed_by="system",
        notes="Initial deployment"
    )
    
    print("\nLineage:")
    print(f"  Dataset Checksum: {lineage.dataset_checksum}")
    print(f"  Transformations: {len(lineage.transformations)}")
    print(f"  Deployment History: {len(lineage.deployment_history)}")
    
    # Convert to dict and back
    lineage_dict = lineage.to_dict()
    lineage2 = ModelLineage.from_dict(lineage_dict)
    
    print(f"\nLineage serialized and deserialized successfully: {lineage.to_dict() == lineage2.to_dict()}")
