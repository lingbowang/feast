"""
Unit tests for FeatureStore.get_latest_features() method.

Tests the public API for exporting the latest feature values per entity from a feature view.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feast import FeatureStore, FeatureView
from feast.data_source import FileSource
from feast.entity import Entity
from feast.errors import FeatureViewNotFoundException
from feast.feature_view import Field
from feast.infra.offline_stores.offline_store import RetrievalJob
from feast.repo_config import RepoConfig
from feast.types import Float32, Int64


def test_get_latest_features_basic():
    """Test that get_latest_features calls pull_latest_from_table_or_query correctly."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create a minimal feature store config
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        # Create a feature store
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        # Create test entities and feature view
        entity = Entity(name="driver_id", join_keys=["driver_id"])
        
        batch_source = FileSource(
            path=str(repo_path / "data.parquet"),
            timestamp_field="event_timestamp",
        )
        
        feature_view = FeatureView(
            name="driver_stats",
            entities=[entity],
            ttl=timedelta(days=7),
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="acc_rate", dtype=Float32),
                Field(name="driver_id", dtype=Int64),
            ],
            source=batch_source,
        )
        
        # Apply entities and feature views
        fs.apply([entity, feature_view])
        
        # Mock the provider's offline store
        mock_job = MagicMock(spec=RetrievalJob)
        mock_job.to_df.return_value = pd.DataFrame({
            "driver_id": [1, 2],
            "conv_rate": [0.9, 0.85],
            "acc_rate": [0.95, 0.92],
            "event_timestamp": [datetime.now(), datetime.now()]
        })
        
        with patch.object(
            fs._get_provider().offline_store,
            "pull_latest_from_table_or_query",
            return_value=mock_job
        ) as mock_pull:
            # Call get_latest_features
            job = fs.get_latest_features(feature_view="driver_stats")
            
            # Verify the method was called
            assert mock_pull.called
            assert isinstance(job, RetrievalJob)
            
            # Verify call arguments
            call_args = mock_pull.call_args
            assert call_args.kwargs["data_source"] == feature_view.batch_source
            assert call_args.kwargs["join_key_columns"] == ["driver_id"]
            assert "conv_rate" in call_args.kwargs["feature_name_columns"]
            assert "acc_rate" in call_args.kwargs["feature_name_columns"]


def test_get_latest_features_with_time_range():
    """Test that get_latest_features respects custom time ranges."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        entity = Entity(name="driver_id", join_keys=["driver_id"])
        
        batch_source = FileSource(
            path=str(repo_path / "data.parquet"),
            timestamp_field="event_timestamp",
        )
        
        feature_view = FeatureView(
            name="driver_stats",
            entities=[entity],
            ttl=timedelta(days=7),
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="driver_id", dtype=Int64),
            ],
            source=batch_source,
        )
        
        fs.apply([entity, feature_view])
        
        mock_job = MagicMock(spec=RetrievalJob)
        
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 7, tzinfo=timezone.utc)
        
        with patch.object(
            fs._get_provider().offline_store,
            "pull_latest_from_table_or_query",
            return_value=mock_job
        ) as mock_pull:
            job = fs.get_latest_features(
                feature_view="driver_stats",
                start_date=start,
                end_date=end,
            )
            
            assert mock_pull.called
            call_args = mock_pull.call_args
            assert call_args.kwargs["start_date"] == start
            assert call_args.kwargs["end_date"] == end


def test_get_latest_features_with_ttl_default():
    """Test that get_latest_features defaults start_date based on TTL."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        entity = Entity(name="driver_id", join_keys=["driver_id"])
        
        batch_source = FileSource(
            path=str(repo_path / "data.parquet"),
            timestamp_field="event_timestamp",
        )
        
        # Feature view with 1 day TTL
        feature_view = FeatureView(
            name="driver_stats",
            entities=[entity],
            ttl=timedelta(days=1),
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="driver_id", dtype=Int64),
            ],
            source=batch_source,
        )
        
        fs.apply([entity, feature_view])
        
        mock_job = MagicMock(spec=RetrievalJob)
        
        with patch.object(
            fs._get_provider().offline_store,
            "pull_latest_from_table_or_query",
            return_value=mock_job
        ) as mock_pull:
            job = fs.get_latest_features(feature_view="driver_stats")
            
            assert mock_pull.called
            call_args = mock_pull.call_args
            
            # Verify that start_date was set based on TTL
            start_date = call_args.kwargs["start_date"]
            end_date = call_args.kwargs["end_date"]
            
            # The difference should be approximately 1 day (the TTL)
            time_diff = end_date - start_date
            assert abs(time_diff.total_seconds() - 86400) < 60  # Within 1 minute tolerance


def test_get_latest_features_no_ttl_uses_large_window():
    """Test that get_latest_features uses a large window when TTL is 0."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        entity = Entity(name="driver_id", join_keys=["driver_id"])
        
        batch_source = FileSource(
            path=str(repo_path / "data.parquet"),
            timestamp_field="event_timestamp",
        )
        
        # Feature view with TTL of 0
        feature_view = FeatureView(
            name="driver_stats",
            entities=[entity],
            ttl=timedelta(seconds=0),
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="driver_id", dtype=Int64),
            ],
            source=batch_source,
        )
        
        fs.apply([entity, feature_view])
        
        mock_job = MagicMock(spec=RetrievalJob)
        
        with patch.object(
            fs._get_provider().offline_store,
            "pull_latest_from_table_or_query",
            return_value=mock_job
        ) as mock_pull:
            job = fs.get_latest_features(feature_view="driver_stats")
            
            assert mock_pull.called
            call_args = mock_pull.call_args
            
            # Verify that start_date was set to a large window (365 days)
            start_date = call_args.kwargs["start_date"]
            end_date = call_args.kwargs["end_date"]
            
            time_diff = end_date - start_date
            # Should be approximately 365 days
            assert abs(time_diff.total_seconds() - (365 * 86400)) < 60


def test_get_latest_features_nonexistent_feature_view():
    """Test that get_latest_features raises error for nonexistent feature view."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        with pytest.raises(FeatureViewNotFoundException):
            fs.get_latest_features(feature_view="nonexistent_view")


def test_get_latest_features_no_batch_source():
    """Test that get_latest_features raises error when feature view has no batch source."""
    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        config = RepoConfig(
            project="test_project",
            registry=str(repo_path / "registry.db"),
            provider="local",
            offline_store={"type": "file"},
        )
        
        fs = FeatureStore(config=config, repo_path=repo_path)
        
        entity = Entity(name="driver_id", join_keys=["driver_id"])
        
        # Feature view without batch source
        feature_view = FeatureView(
            name="driver_stats",
            entities=[entity],
            ttl=timedelta(days=1),
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="driver_id", dtype=Int64),
            ],
            source=None,
        )
        
        fs.apply([entity, feature_view])
        
        with pytest.raises(ValueError, match="no batch source configured"):
            fs.get_latest_features(feature_view="driver_stats")
