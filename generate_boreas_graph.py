#!/usr/bin/env python3
"""
Generate Semantic Forest from Boreas Dataset

This script creates topological graphs from Boreas dataset sequences
and optionally generates hierarchical semantic forests using the
SpatialRelationshipExtractor.
"""

import networkx as nx
import logging
import asyncio
import os
import argparse
from tqdm import tqdm
from datetime import datetime

from embodied_nav.boreas_loader import BoreasLoader
from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
from embodied_nav.llm import LLMInterface

# Define paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_GRAPHS_DIR = os.path.join(PROJECT_ROOT, "semantic_graphs")


async def generate_semantic_forest_from_graph(
    initial_graph: nx.Graph,
    output_file: str,
) -> nx.Graph:
    """
    Generate enhanced semantic forest from an initial graph.

    Args:
        initial_graph: Initial topological graph
        output_file: Path to save the enhanced graph

    Returns:
        Enhanced graph with hierarchical clusters
    """
    print("\n=== Starting Semantic Forest Generation ===")

    # Get non-vehicle nodes (objects) for clustering
    objects = [
        {'id': node, **{k: v for k, v in data.items() if k != 'level'}}
        for node, data in initial_graph.nodes(data=True)
        if data.get('type') not in ('drone', 'vehicle')
    ]

    if not objects:
        # If no objects, use vehicle nodes for spatial clustering
        print("No object nodes found, using vehicle trajectory for clustering...")
        objects = [
            {'id': node, **{k: v for k, v in data.items() if k != 'level'}}
            for node, data in initial_graph.nodes(data=True)
            if data.get('type') in ('drone', 'vehicle')
        ]

    if not objects:
        print("No nodes to cluster. Saving original graph.")
        nx.write_gml(initial_graph, output_file)
        return initial_graph

    print(f"\nProcessing {len(objects)} nodes for clustering...")

    # Initialize LLM interface and relationship extractor
    llm_interface = LLMInterface()
    relationship_extractor = SpatialRelationshipExtractor(llm_interface)

    # Extract spatial relationships and create hierarchical clusters
    print("Extracting spatial relationships and creating clusters...")
    enhanced_graph = await relationship_extractor.extract_relationships(objects)

    # Merge original and enhanced graphs
    print("\nMerging graphs...")
    merged_graph = nx.Graph()

    # Copy graph attributes
    merged_graph.graph.update(initial_graph.graph)
    merged_graph.graph['enhanced'] = True
    merged_graph.graph['enhanced_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add nodes from both graphs
    print("Adding nodes...")
    for node, data in tqdm(initial_graph.nodes(data=True), desc="Original nodes"):
        merged_graph.add_node(node, **data)

    for node, data in tqdm(enhanced_graph.nodes(data=True), desc="Enhanced nodes"):
        if node in merged_graph:
            merged_graph.nodes[node].update(data)
        else:
            merged_graph.add_node(node, **data)

    # Add edges from both graphs
    print("Adding edges...")
    total_edges = len(initial_graph.edges()) + len(enhanced_graph.edges())
    with tqdm(total=total_edges, desc="Adding edges") as pbar:
        for u, v, data in initial_graph.edges(data=True):
            merged_graph.add_edge(u, v, **data)
            pbar.update(1)

        for u, v, data in enhanced_graph.edges(data=True):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, **data)
            pbar.update(1)

    # Save merged graph
    print("\nSaving enhanced graph...")
    nx.write_gml(merged_graph, output_file)

    print("\n=== Semantic Forest Generation Complete ===")
    print(f"Enhanced graph saved to: {output_file}")

    return merged_graph


def process_boreas_sequence(
    sequence_id: str,
    boreas_root: str = None,
    sensor: str = "camera",
    sample_interval: int = 10,
    min_distance: float = 2.0,
    generate_forest: bool = True,
    include_objects: bool = False,
) -> tuple:
    """
    Process a Boreas sequence and optionally generate semantic forest.

    Args:
        sequence_id: Boreas sequence ID
        boreas_root: Path to Boreas dataset root
        sensor: Sensor type for poses
        sample_interval: Frame sampling interval
        min_distance: Minimum distance between nodes
        generate_forest: Whether to generate hierarchical clusters
        include_objects: Include object labels if available

    Returns:
        Tuple of (graph, output_path)
    """
    print(f"\n=== Processing Boreas Sequence: {sequence_id} ===")

    # Initialize loader
    loader = BoreasLoader(
        boreas_root=boreas_root,
        sample_interval=sample_interval,
        min_distance=min_distance,
    )

    # Create initial graph
    print("\nCreating topological graph from trajectory...")
    initial_graph, initial_path = loader.process_sequence(
        sequence_id,
        sensor=sensor,
        include_objects=include_objects,
        save=True,
    )

    if initial_graph.number_of_nodes() == 0:
        print("Error: Empty graph generated")
        return None, None

    print(f"Initial graph: {initial_graph.number_of_nodes()} nodes, {initial_graph.number_of_edges()} edges")

    if not generate_forest:
        return initial_graph, initial_path

    # Generate semantic forest
    enhanced_path = initial_path.replace('.gml', '_enhanced.gml')
    enhanced_graph = asyncio.run(
        generate_semantic_forest_from_graph(initial_graph, enhanced_path)
    )

    return enhanced_graph, enhanced_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate graphs from Boreas dataset"
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
        help="Sequence ID to process (e.g., boreas-2020-11-26-13-58)"
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
        "--no-forest",
        action="store_true",
        help="Skip semantic forest generation (only create trajectory graph)"
    )
    parser.add_argument(
        "--include-objects",
        action="store_true",
        help="Include object labels if available"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sequences"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available sequences"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize loader to list sequences
    loader = BoreasLoader(boreas_root=args.boreas_root)
    sequences = loader.list_sequences()

    if args.list:
        print(f"\nAvailable sequences ({len(sequences)}):")
        for seq in sequences:
            print(f"  - {seq}")
        return

    if args.all:
        # Process all sequences
        print(f"\nProcessing all {len(sequences)} sequences...")
        for seq_id in sequences:
            try:
                process_boreas_sequence(
                    seq_id,
                    boreas_root=args.boreas_root,
                    sensor=args.sensor,
                    sample_interval=args.sample_interval,
                    min_distance=args.min_distance,
                    generate_forest=not args.no_forest,
                    include_objects=args.include_objects,
                )
            except Exception as e:
                logging.error(f"Error processing {seq_id}: {e}")
                continue
        return

    # Process single sequence
    if args.sequence is None:
        if sequences:
            print(f"\nNo sequence specified. Using first available: {sequences[0]}")
            args.sequence = sequences[0]
        else:
            print("Error: No sequences found in dataset")
            return

    try:
        graph, output_path = process_boreas_sequence(
            args.sequence,
            boreas_root=args.boreas_root,
            sensor=args.sensor,
            sample_interval=args.sample_interval,
            min_distance=args.min_distance,
            generate_forest=not args.no_forest,
            include_objects=args.include_objects,
        )

        if output_path:
            print(f"\n=== Complete ===")
            print(f"Output: {output_path}")
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")

    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
