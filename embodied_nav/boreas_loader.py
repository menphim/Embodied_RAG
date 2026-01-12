"""
Boreas Dataset Loader for Embodied RAG

This module provides functionality to load Boreas dataset and create
topological graphs compatible with the Embodied RAG system.
Uses GNSS coordinates (UTM) for consistent global positioning across sequences.
"""

import os
import sys
import logging
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add pyboreas to path if needed
BOREAS_ROOT = Path(__file__).parent.parent / "data" / "boreas"
PYBOREAS_PATH = BOREAS_ROOT / "pyboreas"
if str(PYBOREAS_PATH) not in sys.path:
    sys.path.insert(0, str(PYBOREAS_PATH))

from pyboreas import BoreasDataset
from pyboreas.data.sequence import Sequence

# UTM conversion
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not installed. Install with: uv pip install pyproj")


class GNSSConverter:
    """Convert GNSS coordinates (lat/lon) to UTM coordinates."""

    # Boreas dataset is in Toronto area - UTM Zone 17N
    UTM_ZONE = 17
    UTM_EPSG = 32617  # WGS84 / UTM zone 17N

    def __init__(self):
        if not HAS_PYPROJ:
            raise ImportError("pyproj is required for GNSS conversion")
        # WGS84 (lat/lon) to UTM zone 17N
        self.transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84
            f"EPSG:{self.UTM_EPSG}",  # UTM zone 17N
            always_xy=True  # lon, lat order
        )

    def to_utm(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        Convert latitude/longitude to UTM coordinates.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees

        Returns:
            Tuple of (easting, northing) in meters
        """
        # pyproj expects (lon, lat) order with always_xy=True
        easting, northing = self.transformer.transform(longitude, latitude)
        return float(easting), float(northing)


class BoreasLoader:
    """
    Loader for Boreas dataset that creates topological graphs
    compatible with the Embodied RAG system.
    Uses GNSS (UTM) coordinates for global positioning.
    """

    def __init__(
        self,
        boreas_root: str = None,
        output_dir: str = None,
        sample_interval: int = 10,
        min_distance: float = 2.0,
        use_gnss: bool = True,
    ):
        """
        Initialize the Boreas loader.

        Args:
            boreas_root: Path to Boreas dataset root
            output_dir: Directory to save generated graphs
            sample_interval: Sample every N frames (to reduce density)
            min_distance: Minimum distance between nodes in meters
            use_gnss: Use GNSS (UTM) coordinates instead of local ENU
        """
        self.boreas_root = boreas_root or str(BOREAS_ROOT)
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "semantic_graphs"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_interval = sample_interval
        self.min_distance = min_distance
        self.use_gnss = use_gnss and HAS_PYPROJ

        if self.use_gnss:
            self.gnss_converter = GNSSConverter()
        else:
            self.gnss_converter = None
            if use_gnss and not HAS_PYPROJ:
                logging.warning("pyproj not available, falling back to local ENU coordinates")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_sequence(self, sequence_id: str) -> Optional[Sequence]:
        """
        Load a single Boreas sequence.

        Args:
            sequence_id: Sequence ID (e.g., 'boreas-2020-11-26-13-58')

        Returns:
            Sequence object or None if not found
        """
        try:
            seq_path = os.path.join(self.boreas_root, sequence_id)
            if not os.path.exists(seq_path):
                self.logger.error(f"Sequence not found: {seq_path}")
                return None

            bd = BoreasDataset(self.boreas_root, split=[[sequence_id]])
            if bd.sequences:
                return bd.sequences[0]
            return None
        except Exception as e:
            self.logger.error(f"Error loading sequence {sequence_id}: {e}")
            return None

    def list_sequences(self) -> List[str]:
        """List all available sequences in the dataset."""
        sequences = []
        for item in os.listdir(self.boreas_root):
            if item.startswith("boreas-") and os.path.isdir(
                os.path.join(self.boreas_root, item)
            ):
                # Check if it has required data
                applanix_path = os.path.join(self.boreas_root, item, "applanix")
                if os.path.exists(applanix_path):
                    sequences.append(item)
        return sorted(sequences)

    def _load_gps_data(self, sequence_id: str) -> Optional[pd.DataFrame]:
        """
        Load GPS post-process data for a sequence.

        Args:
            sequence_id: Sequence ID

        Returns:
            DataFrame with GPS data or None
        """
        gps_file = os.path.join(
            self.boreas_root, sequence_id, "applanix", "gps_post_process.csv"
        )
        if not os.path.exists(gps_file):
            self.logger.warning(f"GPS file not found: {gps_file}")
            return None

        try:
            df = pd.read_csv(gps_file)
            return df
        except Exception as e:
            self.logger.error(f"Error loading GPS data: {e}")
            return None

    def _get_gnss_position(
        self,
        gps_df: pd.DataFrame,
        timestamp: float
    ) -> Optional[Dict[str, float]]:
        """
        Get GNSS position (UTM) for a given timestamp.

        Args:
            gps_df: GPS DataFrame
            timestamp: Timestamp to lookup

        Returns:
            Position dictionary with x, y, z (UTM coordinates)
        """
        if gps_df is None or len(gps_df) == 0:
            return None

        # Find closest GPS timestamp
        idx = (gps_df['GPSTime'] - timestamp).abs().idxmin()
        row = gps_df.loc[idx]

        # Convert lat/lon to UTM
        lat = row['latitude']
        lon = row['longitude']
        alt = row['altitude']

        # Check for valid coordinates (Boreas uses radians in some files)
        # Toronto is around 43.7°N, -79.4°W
        if abs(lat) < 2:  # Likely radians
            lat = np.degrees(lat)
            lon = np.degrees(lon)

        easting, northing = self.gnss_converter.to_utm(lat, lon)

        return {
            'x': easting,
            'y': northing,
            'z': float(alt),
            'latitude': float(lat),
            'longitude': float(lon),
        }

    def _pose_to_position(self, pose: np.ndarray) -> Dict[str, float]:
        """
        Convert 4x4 pose matrix to position dictionary (local ENU).

        Args:
            pose: 4x4 homogeneous transformation matrix

        Returns:
            Dictionary with x, y, z coordinates
        """
        return {
            'x': float(pose[0, 3]),
            'y': float(pose[1, 3]),
            'z': float(pose[2, 3])
        }

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate Euclidean distance between two positions."""
        return float(np.sqrt(
            (pos1['x'] - pos2['x']) ** 2 +
            (pos1['y'] - pos2['y']) ** 2 +
            (pos1['z'] - pos2['z']) ** 2
        ))

    def _pose_to_yaw(self, pose: np.ndarray) -> float:
        """Extract yaw angle from pose matrix."""
        R = pose[:3, :3]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return float(yaw)

    def create_graph_from_sequence(
        self,
        sequence: Sequence,
        sensor: str = "camera",
        include_objects: bool = False,
    ) -> nx.Graph:
        """
        Create a topological graph from a Boreas sequence.

        Args:
            sequence: Boreas Sequence object
            sensor: Sensor to use for poses ('camera', 'lidar', 'radar')
            include_objects: Whether to include detected objects (requires labels)

        Returns:
            NetworkX graph with vehicle trajectory nodes
        """
        G = nx.Graph()
        G.graph['environment'] = sequence.ID
        G.graph['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        G.graph['sensor'] = sensor
        G.graph['source'] = 'boreas'
        G.graph['coordinate_system'] = 'UTM_17N' if self.use_gnss else 'ENU_local'

        # Get frames based on sensor type
        if sensor == "camera":
            frames = sequence.camera_frames
        elif sensor == "lidar":
            frames = sequence.lidar_frames
        elif sensor == "radar":
            frames = sequence.radar_frames
        else:
            raise ValueError(f"Unknown sensor type: {sensor}")

        if not frames:
            self.logger.warning(f"No {sensor} frames found in sequence")
            return G

        self.logger.info(f"Processing {len(frames)} {sensor} frames")

        # Load GPS data if using GNSS coordinates
        gps_df = None
        if self.use_gnss:
            gps_df = self._load_gps_data(sequence.ID)
            if gps_df is None:
                self.logger.warning("Falling back to local ENU coordinates")

        # Sample frames and create nodes
        last_position = None
        last_node_id = None
        node_count = 0

        for i, frame in enumerate(frames):
            # Sample every N frames
            if i % self.sample_interval != 0:
                continue

            # Get pose (4x4 transformation matrix)
            pose = frame.pose
            if pose is None:
                continue

            # Get position based on coordinate system
            if self.use_gnss and gps_df is not None:
                position = self._get_gnss_position(gps_df, frame.timestamp)
                if position is None:
                    position = self._pose_to_position(pose)
            else:
                position = self._pose_to_position(pose)

            # Check minimum distance
            if last_position is not None:
                distance = self._calculate_distance(position, last_position)
                if distance < self.min_distance:
                    continue

            # Create vehicle node
            node_id = f"vehicle_{node_count}"
            yaw = self._pose_to_yaw(pose)

            node_attrs = {
                'position_x': position['x'],
                'position_y': position['y'],
                'position_z': position['z'],
                'yaw': yaw,
                'type': 'vehicle',
                'level': 0,
                'timestamp': float(frame.timestamp),
                'frame_id': frame.frame,
                'sequence_id': sequence.ID,
            }

            # Add lat/lon if available
            if 'latitude' in position:
                node_attrs['latitude'] = position['latitude']
                node_attrs['longitude'] = position['longitude']

            G.add_node(node_id, **node_attrs)

            # Connect to previous node
            if last_node_id is not None:
                edge_distance = self._calculate_distance(position, last_position)
                G.add_edge(
                    last_node_id,
                    node_id,
                    distance=edge_distance,
                    type='vehicle_path'
                )

            last_position = position
            last_node_id = node_id
            node_count += 1

        self.logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Add objects if available and requested
        if include_objects:
            self._add_objects_from_labels(G, sequence, gps_df)

        return G

    def _add_objects_from_labels(
        self,
        G: nx.Graph,
        sequence: Sequence,
        gps_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Add object nodes from label files if available.

        Args:
            G: Graph to add objects to
            sequence: Boreas sequence
            gps_df: GPS DataFrame for coordinate conversion
        """
        label_count = 0
        seen_objects = set()

        for frame in sequence.lidar_frames:
            if not frame.has_bbs():
                continue

            try:
                bbs = frame.get_bounding_boxes(
                    sequence.labelFiles,
                    sequence.labelTimes,
                    sequence.labelPoses
                )

                if bbs is None:
                    continue

                for bb in bbs.bbs:
                    object_id = f"{bb.label}_{bb.uuid}"

                    if object_id in seen_objects:
                        continue
                    seen_objects.add(object_id)

                    # Get position in global frame
                    pos_local = bb.pos.flatten()
                    pos_global = frame.pose @ np.array([pos_local[0], pos_local[1], pos_local[2], 1])

                    # Convert to UTM if using GNSS
                    if self.use_gnss and gps_df is not None:
                        # Get vehicle GNSS position and add local offset
                        vehicle_gnss = self._get_gnss_position(gps_df, frame.timestamp)
                        if vehicle_gnss:
                            # Add local offset to GNSS position
                            position_x = vehicle_gnss['x'] + float(pos_global[0])
                            position_y = vehicle_gnss['y'] + float(pos_global[1])
                            position_z = float(pos_global[2])
                        else:
                            position_x = float(pos_global[0])
                            position_y = float(pos_global[1])
                            position_z = float(pos_global[2])
                    else:
                        position_x = float(pos_global[0])
                        position_y = float(pos_global[1])
                        position_z = float(pos_global[2])

                    G.add_node(
                        object_id,
                        position_x=position_x,
                        position_y=position_y,
                        position_z=position_z,
                        type='object',
                        label=bb.label,
                        uuid=bb.uuid,
                        extent_width=float(bb.extent[0, 0]),
                        extent_length=float(bb.extent[1, 0]),
                        extent_height=float(bb.extent[2, 0]),
                        level=0,
                        sequence_id=sequence.ID,
                    )
                    label_count += 1

            except Exception as e:
                self.logger.debug(f"Error processing labels for frame {frame.frame}: {e}")
                continue

        if label_count > 0:
            self.logger.info(f"Added {label_count} object nodes from labels")

    def save_graph(self, G: nx.Graph, filename: str = None) -> str:
        """
        Save graph to GML file.

        Args:
            G: Graph to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            env = G.graph.get('environment', 'boreas')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"boreas_{env}_{timestamp}.gml"

        filepath = os.path.join(self.output_dir, filename)
        nx.write_gml(G, filepath)
        self.logger.info(f"Graph saved to {filepath}")
        return filepath

    def process_sequence(
        self,
        sequence_id: str,
        sensor: str = "camera",
        include_objects: bool = False,
        save: bool = True,
    ) -> Tuple[nx.Graph, Optional[str]]:
        """
        Process a single sequence and optionally save the graph.

        Args:
            sequence_id: Sequence ID
            sensor: Sensor type for poses
            include_objects: Include object labels if available
            save: Whether to save the graph

        Returns:
            Tuple of (graph, filepath or None)
        """
        self.logger.info(f"Processing sequence: {sequence_id}")

        sequence = self.load_sequence(sequence_id)
        if sequence is None:
            return nx.Graph(), None

        G = self.create_graph_from_sequence(
            sequence,
            sensor=sensor,
            include_objects=include_objects
        )

        filepath = None
        if save and G.number_of_nodes() > 0:
            filepath = self.save_graph(G)

        return G, filepath

    def process_multiple_sequences(
        self,
        sequence_ids: List[str] = None,
        sensor: str = "camera",
        include_objects: bool = False,
        save: bool = True,
    ) -> Tuple[nx.Graph, Optional[str]]:
        """
        Process multiple sequences and merge into a single graph.
        Requires GNSS coordinates for proper alignment.

        Args:
            sequence_ids: List of sequence IDs (None = all available)
            sensor: Sensor type for poses
            include_objects: Include object labels if available
            save: Whether to save the merged graph

        Returns:
            Tuple of (merged graph, filepath or None)
        """
        if not self.use_gnss:
            self.logger.warning(
                "Merging multiple sequences without GNSS coordinates may result in "
                "misaligned data. Consider enabling GNSS mode."
            )

        if sequence_ids is None:
            sequence_ids = self.list_sequences()

        self.logger.info(f"Processing {len(sequence_ids)} sequences...")

        merged_graph = nx.Graph()
        merged_graph.graph['source'] = 'boreas'
        merged_graph.graph['coordinate_system'] = 'UTM_17N' if self.use_gnss else 'ENU_local'
        merged_graph.graph['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_graph.graph['sequences'] = ','.join(sequence_ids)

        total_nodes = 0
        for seq_id in sequence_ids:
            try:
                G, _ = self.process_sequence(
                    seq_id,
                    sensor=sensor,
                    include_objects=include_objects,
                    save=False,
                )

                if G.number_of_nodes() == 0:
                    continue

                # Add nodes with sequence prefix to avoid ID conflicts
                for node, data in G.nodes(data=True):
                    new_node_id = f"{seq_id}_{node}"
                    merged_graph.add_node(new_node_id, **data)

                # Add edges with updated node IDs
                for u, v, data in G.edges(data=True):
                    new_u = f"{seq_id}_{u}"
                    new_v = f"{seq_id}_{v}"
                    merged_graph.add_edge(new_u, new_v, **data)

                total_nodes += G.number_of_nodes()
                self.logger.info(f"  {seq_id}: {G.number_of_nodes()} nodes added")

            except Exception as e:
                self.logger.error(f"Error processing {seq_id}: {e}")
                continue

        self.logger.info(
            f"Merged graph: {merged_graph.number_of_nodes()} nodes, "
            f"{merged_graph.number_of_edges()} edges"
        )

        filepath = None
        if save and merged_graph.number_of_nodes() > 0:
            filename = f"boreas_merged_{len(sequence_ids)}seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gml"
            filepath = self.save_graph(merged_graph, filename)

        return merged_graph, filepath


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create topological graphs from Boreas dataset"
    )
    parser.add_argument(
        "--boreas-root",
        type=str,
        default=None,
        help="Path to Boreas dataset root"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Specific sequence ID to process (default: list available)"
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="camera",
        choices=["camera", "lidar", "radar"],
        help="Sensor to use for poses"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Sample every N frames"
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=2.0,
        help="Minimum distance between nodes (meters)"
    )
    parser.add_argument(
        "--include-objects",
        action="store_true",
        help="Include object labels if available"
    )
    parser.add_argument(
        "--no-gnss",
        action="store_true",
        help="Use local ENU coordinates instead of GNSS (UTM)"
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge all sequences into one graph"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sequences and exit"
    )

    args = parser.parse_args()

    loader = BoreasLoader(
        boreas_root=args.boreas_root,
        sample_interval=args.sample_interval,
        min_distance=args.min_distance,
        use_gnss=not args.no_gnss,
    )

    if args.list:
        sequences = loader.list_sequences()
        print(f"\nAvailable sequences ({len(sequences)}):")
        for seq in sequences:
            print(f"  - {seq}")
        return

    if args.merge_all:
        G, filepath = loader.process_multiple_sequences(
            sensor=args.sensor,
            include_objects=args.include_objects,
        )
        if filepath:
            print(f"\nMerged graph saved to: {filepath}")
            print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return

    if args.sequence is None:
        sequences = loader.list_sequences()
        print(f"\nAvailable sequences ({len(sequences)}):")
        for seq in sequences:
            print(f"  - {seq}")
        if sequences:
            print(f"\nProcessing first sequence: {sequences[0]}")
            args.sequence = sequences[0]
        else:
            return

    G, filepath = loader.process_sequence(
        args.sequence,
        sensor=args.sensor,
        include_objects=args.include_objects,
    )

    if filepath:
        print(f"\nGraph saved to: {filepath}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
