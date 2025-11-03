"""
Smart Grid Communication Network Simulator with QoS-Aware Resource Allocation
Demonstrates telecom optimization for critical energy infrastructure

Key Features:
- Multi-tier network topology (AMI, distribution automation, SCADA)
- Traffic classification with strict latency/reliability requirements
- Dynamic bandwidth allocation using optimization algorithms
- Network congestion handling and priority scheduling
- Performance metrics and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum
import heapq
from collections import defaultdict
import json


class TrafficType(Enum):
    """Smart grid communication traffic categories"""
    SCADA_CONTROL = 1      # Critical: Grid control commands
    PROTECTION = 2          # Critical: Fault detection/isolation
    AMI_URGENT = 3         # High: Demand response signals
    AMI_PERIODIC = 4       # Medium: Regular meter readings
    VIDEO_MONITOR = 5      # Low: Substation surveillance
    
    
@dataclass
class QoSRequirement:
    """Quality of Service parameters for each traffic type"""
    max_latency_ms: float
    min_reliability: float
    priority: int
    bandwidth_kbps: float
    

class QoSProfile:
    """Predefined QoS profiles based on IEC 61850 and smart grid standards"""
    PROFILES = {
        TrafficType.SCADA_CONTROL: QoSRequirement(
            max_latency_ms=10,      # Type 1A: Trip signals
            min_reliability=0.9999,
            priority=1,
            bandwidth_kbps=64
        ),
        TrafficType.PROTECTION: QoSRequirement(
            max_latency_ms=3,       # Type P: Protection
            min_reliability=0.99999,
            priority=0,
            bandwidth_kbps=128
        ),
        TrafficType.AMI_URGENT: QoSRequirement(
            max_latency_ms=100,     # Demand response
            min_reliability=0.999,
            priority=2,
            bandwidth_kbps=32
        ),
        TrafficType.AMI_PERIODIC: QoSRequirement(
            max_latency_ms=2000,    # 15-min interval data
            min_reliability=0.99,
            priority=4,
            bandwidth_kbps=16
        ),
        TrafficType.VIDEO_MONITOR: QoSRequirement(
            max_latency_ms=500,
            min_reliability=0.95,
            priority=5,
            bandwidth_kbps=512
        )
    }


@dataclass
class Packet:
    """Network packet representation"""
    packet_id: int
    traffic_type: TrafficType
    source_node: int
    dest_node: int
    size_bytes: int
    generation_time: float
    deadline: float
    priority: int
    hop_count: int = 0
    
    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class NetworkNode:
    """Represents a smart grid device (smart meter, RTU, concentrator, etc.)"""
    node_id: int
    node_type: str  # 'meter', 'concentrator', 'substation', 'control_center'
    processing_capacity: float  # packets/ms
    buffer_size: int
    queue: List[Packet] = field(default_factory=list)
    
    def enqueue(self, packet: Packet) -> bool:
        """Add packet to queue if space available"""
        if len(self.queue) < self.buffer_size:
            heapq.heappush(self.queue, packet)
            return True
        return False
    
    def dequeue(self) -> Packet:
        """Remove highest priority packet"""
        return heapq.heappop(self.queue) if self.queue else None


@dataclass
class NetworkLink:
    """Communication link between nodes"""
    link_id: int
    source: int
    destination: int
    bandwidth_mbps: float
    latency_ms: float
    utilization: float = 0.0
    
    def available_bandwidth(self) -> float:
        """Return unused bandwidth"""
        return self.bandwidth_mbps * (1 - self.utilization)


class SmartGridNetwork:
    """Main network simulator"""
    
    def __init__(self, topology_type: str = 'hierarchical'):
        self.nodes: Dict[int, NetworkNode] = {}
        self.links: Dict[int, NetworkLink] = {}
        self.routing_table: Dict[Tuple[int, int], List[int]] = {}
        self.packet_counter = 0
        self.current_time = 0.0
        
        # Statistics
        self.delivered_packets = []
        self.dropped_packets = []
        self.latency_stats = defaultdict(list)
        
        self._build_topology(topology_type)
        self._compute_routing()
        
    def _build_topology(self, topology_type: str):
        """Create network topology"""
        if topology_type == 'hierarchical':
            # Layer 1: Smart meters (AMI)
            for i in range(20):
                self.add_node(NetworkNode(i, 'meter', 0.1, 10))
            
            # Layer 2: Concentrators
            for i in range(20, 25):
                self.add_node(NetworkNode(i, 'concentrator', 1.0, 50))
            
            # Layer 3: Substation RTUs
            for i in range(25, 27):
                self.add_node(NetworkNode(i, 'substation', 5.0, 100))
            
            # Layer 4: Control center
            self.add_node(NetworkNode(27, 'control_center', 10.0, 200))
            
            # Create links: meters -> concentrators
            link_id = 0
            for meter in range(20):
                concentrator = 20 + (meter % 5)
                self.add_link(NetworkLink(link_id, meter, concentrator, 
                                         1.0, 5.0))  # 1 Mbps, 5ms
                link_id += 1
            
            # concentrators -> substations
            for conc in range(20, 25):
                substation = 25 + (conc % 2)
                self.add_link(NetworkLink(link_id, conc, substation, 
                                         10.0, 2.0))  # 10 Mbps, 2ms
                link_id += 1
            
            # substations -> control center
            for sub in range(25, 27):
                self.add_link(NetworkLink(link_id, sub, 27, 
                                         100.0, 1.0))  # 100 Mbps, 1ms
                link_id += 1
    
    def add_node(self, node: NetworkNode):
        self.nodes[node.node_id] = node
    
    def add_link(self, link: NetworkLink):
        self.links[link.link_id] = link
    
    def _compute_routing(self):
        """Simple shortest path routing using Dijkstra"""
        for source in self.nodes:
            distances = {node: float('inf') for node in self.nodes}
            distances[source] = 0
            previous = {node: None for node in self.nodes}
            unvisited = set(self.nodes.keys())
            
            while unvisited:
                current = min(unvisited, key=lambda x: distances[x])
                unvisited.remove(current)
                
                # Find neighbors
                for link in self.links.values():
                    if link.source == current and link.destination in unvisited:
                        alt = distances[current] + link.latency_ms
                        if alt < distances[link.destination]:
                            distances[link.destination] = alt
                            previous[link.destination] = current
            
            # Build paths
            for dest in self.nodes:
                if dest != source:
                    path = []
                    current = dest
                    while previous[current] is not None:
                        path.append(current)
                        current = previous[current]
                    path.append(source)
                    self.routing_table[(source, dest)] = list(reversed(path))
    
    def generate_traffic(self, num_packets: int = 100, duration_ms: float = 1000):
        """Generate realistic smart grid traffic patterns"""
        packets = []
        
        for _ in range(num_packets):
            # Traffic distribution based on smart grid characteristics
            rand = np.random.random()
            if rand < 0.05:  # 5% critical control
                traffic_type = TrafficType.SCADA_CONTROL
                source = np.random.choice(range(25, 27))  # From substations
            elif rand < 0.08:  # 3% protection
                traffic_type = TrafficType.PROTECTION
                source = np.random.choice(range(25, 27))
            elif rand < 0.15:  # 7% urgent AMI
                traffic_type = TrafficType.AMI_URGENT
                source = np.random.choice(range(20))
            elif rand < 0.80:  # 65% periodic AMI
                traffic_type = TrafficType.AMI_PERIODIC
                source = np.random.choice(range(20))
            else:  # 20% video
                traffic_type = TrafficType.VIDEO_MONITOR
                source = np.random.choice(range(25, 27))
            
            qos = QoSProfile.PROFILES[traffic_type]
            dest = 27  # Control center
            
            packet = Packet(
                packet_id=self.packet_counter,
                traffic_type=traffic_type,
                source_node=source,
                dest_node=dest,
                size_bytes=int(np.random.uniform(64, 1500)),
                generation_time=np.random.uniform(0, duration_ms),
                deadline=np.random.uniform(0, duration_ms) + qos.max_latency_ms,
                priority=qos.priority
            )
            packets.append(packet)
            self.packet_counter += 1
        
        return sorted(packets, key=lambda p: p.generation_time)
    
    def simulate(self, packets: List[Packet], time_step_ms: float = 0.1):
        """Run network simulation"""
        self.current_time = 0.0
        event_queue = [(p.generation_time, 'generate', p) for p in packets]
        heapq.heapify(event_queue)
        
        while event_queue:
            time, event_type, packet = heapq.heappop(event_queue)
            self.current_time = time
            
            if event_type == 'generate':
                # Packet enters network
                source_node = self.nodes[packet.source_node]
                if source_node.enqueue(packet):
                    # Schedule transmission
                    heapq.heappush(event_queue, 
                                 (time + time_step_ms, 'transmit', packet))
                else:
                    self.dropped_packets.append(packet)
            
            elif event_type == 'transmit':
                if packet.dest_node == packet.source_node:
                    # Packet reached destination
                    latency = self.current_time - packet.generation_time
                    self.delivered_packets.append(packet)
                    self.latency_stats[packet.traffic_type].append(latency)
                else:
                    # Forward to next hop
                    route = self.routing_table.get((packet.source_node, packet.dest_node))
                    if route and len(route) > 1:
                        next_hop = route[1]
                        
                        # Find link
                        link = None
                        for l in self.links.values():
                            if l.source == packet.source_node and l.destination == next_hop:
                                link = l
                                break
                        
                        if link:
                            # Calculate transmission time
                            tx_time = (packet.size_bytes * 8) / (link.bandwidth_mbps * 1000)
                            arrival_time = time + tx_time + link.latency_ms
                            
                            # Update link utilization
                            link.utilization = min(0.95, link.utilization + 0.01)
                            
                            # Check deadline
                            if arrival_time <= packet.deadline:
                                packet.source_node = next_hop
                                packet.hop_count += 1
                                heapq.heappush(event_queue, 
                                             (arrival_time, 'transmit', packet))
                            else:
                                self.dropped_packets.append(packet)
    
    def analyze_performance(self) -> Dict:
        """Calculate network performance metrics"""
        total_packets = len(self.delivered_packets) + len(self.dropped_packets)
        
        results = {
            'total_packets': total_packets,
            'delivered': len(self.delivered_packets),
            'dropped': len(self.dropped_packets),
            'delivery_ratio': len(self.delivered_packets) / total_packets if total_packets > 0 else 0,
            'per_traffic_type': {}
        }
        
        for traffic_type in TrafficType:
            latencies = self.latency_stats[traffic_type]
            qos = QoSProfile.PROFILES[traffic_type]
            
            if latencies:
                met_deadline = sum(1 for l in latencies if l <= qos.max_latency_ms)
                results['per_traffic_type'][traffic_type.name] = {
                    'count': len(latencies),
                    'avg_latency_ms': np.mean(latencies),
                    'max_latency_ms': np.max(latencies),
                    'qos_requirement_ms': qos.max_latency_ms,
                    'qos_compliance': met_deadline / len(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95)
                }
        
        return results
    
    def visualize_results(self, results: Dict):
        """Create performance visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Latency by traffic type
        ax1 = axes[0, 0]
        traffic_types = []
        avg_latencies = []
        qos_requirements = []
        
        for tt_name, stats in results['per_traffic_type'].items():
            traffic_types.append(tt_name.replace('_', '\n'))
            avg_latencies.append(stats['avg_latency_ms'])
            qos_requirements.append(stats['qos_requirement_ms'])
        
        x = np.arange(len(traffic_types))
        width = 0.35
        ax1.bar(x - width/2, avg_latencies, width, label='Actual Latency', color='steelblue')
        ax1.bar(x + width/2, qos_requirements, width, label='QoS Requirement', color='coral')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Average Latency vs QoS Requirements')
        ax1.set_xticks(x)
        ax1.set_xticklabels(traffic_types, fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. QoS Compliance
        ax2 = axes[0, 1]
        compliance_rates = [stats['qos_compliance'] * 100 
                           for stats in results['per_traffic_type'].values()]
        colors = ['green' if c >= 99 else 'orange' if c >= 95 else 'red' 
                 for c in compliance_rates]
        ax2.bar(traffic_types, compliance_rates, color=colors)
        ax2.axhline(y=99, color='r', linestyle='--', label='Target (99%)')
        ax2.set_ylabel('Compliance (%)')
        ax2.set_title('QoS Compliance Rate by Traffic Type')
        ax2.set_ylim([90, 101])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Latency distribution (box plot)
        ax3 = axes[1, 0]
        latency_data = [self.latency_stats[tt] for tt in TrafficType 
                       if self.latency_stats[tt]]
        ax3.boxplot(latency_data, labels=[tt.name.replace('_', '\n') 
                                          for tt in TrafficType if self.latency_stats[tt]])
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency Distribution')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # 4. Overall network performance
        ax4 = axes[1, 1]
        metrics = ['Delivery\nRatio', 'Avg Network\nUtilization']
        values = [results['delivery_ratio'] * 100, 
                 np.mean([link.utilization for link in self.links.values()]) * 100]
        bars = ax4.bar(metrics, values, color=['green', 'steelblue'])
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Overall Network Performance')
        ax4.set_ylim([0, 100])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('smartgrid_network_performance.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'smartgrid_network_performance.png'")
        plt.show()


def main():
    """Run simulation and analysis"""
    print("=" * 70)
    print("Smart Grid Communication Network Simulator")
    print("=" * 70)
    
    # Create network
    network = SmartGridNetwork(topology_type='hierarchical')
    print(f"\nNetwork topology created:")
    print(f"  - Nodes: {len(network.nodes)}")
    print(f"  - Links: {len(network.links)}")
    
    # Generate traffic
    packets = network.generate_traffic(num_packets=500, duration_ms=5000)
    print(f"\nGenerated {len(packets)} packets across {len(TrafficType)} traffic types")
    
    # Run simulation
    print("\nRunning simulation...")
    network.simulate(packets)
    
    # Analyze results
    results = network.analyze_performance()
    
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Performance:")
    print(f"  Total packets: {results['total_packets']}")
    print(f"  Delivered: {results['delivered']}")
    print(f"  Dropped: {results['dropped']}")
    print(f"  Delivery ratio: {results['delivery_ratio']*100:.2f}%")
    
    print("\nPer-Traffic-Type Analysis:")
    print("-" * 70)
    for tt_name, stats in results['per_traffic_type'].items():
        print(f"\n{tt_name}:")
        print(f"  Packets: {stats['count']}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.2f} ms")
        print(f"  QoS requirement: {stats['qos_requirement_ms']:.2f} ms")
        print(f"  QoS compliance: {stats['qos_compliance']*100:.2f}%")
        print(f"  95th percentile: {stats['p95_latency_ms']:.2f} ms")
    
    # Visualize
    print("\nGenerating visualizations...")
    network.visualize_results(results)
    
    # Export results
    with open('simulation_results.json', 'w') as f:
        # Convert enum keys to strings for JSON serialization
        exportable_results = {
            'total_packets': results['total_packets'],
            'delivered': results['delivered'],
            'dropped': results['dropped'],
            'delivery_ratio': results['delivery_ratio'],
            'per_traffic_type': {k: v for k, v in results['per_traffic_type'].items()}
        }
        json.dump(exportable_results, f, indent=2)
    print("Results exported to 'simulation_results.json'")


if __name__ == "__main__":
    main()