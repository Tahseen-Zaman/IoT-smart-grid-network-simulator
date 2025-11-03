# IoT-smart-grid-network-simulator
Smart Grid Communication Network Simulator with Dynamic Resource Allocation
# Smart Grid Communication Network Simulator

## Overview
A Python-based discrete-event simulator for analyzing communication network performance in smart grid infrastructure. Implements QoS-aware packet scheduling, hierarchical network topology, and traffic prioritization based on IEC 61850 standards.

## Key Features

### 1. **Multi-Tier Network Architecture**
- **AMI Layer**: 20 smart meters generating periodic consumption data
- **Aggregation Layer**: 5 data concentrators with buffering
- **Distribution Layer**: 2 substation RTUs handling protection/control
- **Control Layer**: Centralized SCADA/EMS system

### 2. **Traffic Classification**
Implements 5 traffic types with distinct QoS requirements:
- **Protection Signals**: <3ms latency, 99.999% reliability (IEC 61850 Type P)
- **SCADA Control**: <10ms latency, 99.99% reliability (Type 1A)
- **Demand Response**: <100ms latency, 99.9% reliability
- **AMI Periodic**: <2000ms latency, 99% reliability
- **Video Monitoring**: <500ms latency, 95% reliability

### 3. **QoS-Aware Resource Management**
- Priority-based packet scheduling
- Deadline-aware forwarding
- Congestion detection and mitigation
- Buffer overflow handling

### 4. **Performance Metrics**
- End-to-end latency (mean, P95, max)
- QoS compliance rate per traffic class
- Packet delivery ratio
- Network link utilization

## Installation

```bash
# Clone repository
git clone https://github.com/Tahseen-Zaman/smartgrid-network-sim.git
cd smartgrid-network-sim

# Install dependencies
pip install numpy matplotlib

# Run simulation
python smartgrid_simulator.py
```

## Usage

### Basic Simulation
```python
from smartgrid_simulator import SmartGridNetwork

# Initialize network
network = SmartGridNetwork(topology_type='hierarchical')

# Generate traffic (500 packets over 5 seconds)
packets = network.generate_traffic(num_packets=500, duration_ms=5000)

# Run simulation
network.simulate(packets)

# Analyze performance
results = network.analyze_performance()
network.visualize_results(results)
```

### Custom Traffic Generation
```python
# Generate specific traffic mix
packets = []
for i in range(100):
    packet = Packet(
        packet_id=i,
        traffic_type=TrafficType.SCADA_CONTROL,
        source_node=25,
        dest_node=27,
        size_bytes=128,
        generation_time=i * 10,
        deadline=i * 10 + 10,  # 10ms deadline
        priority=1
    )
    packets.append(packet)

network.simulate(packets)
```

## Outputs

1. **Console Report**: Detailed performance statistics
2. **Visualization**: 4-panel figure showing:
   - Latency vs QoS requirements
   - QoS compliance rates
   - Latency distributions (box plots)
   - Overall network metrics
3. **JSON Export**: Machine-readable results file

## Research Extensions

### 1. **Machine Learning-Based QoS Prediction**
Add predictive traffic management:
```python
# Train model on historical traffic patterns
from sklearn.ensemble import RandomForestRegressor

def predict_congestion(network_state):
    features = extract_features(network_state)
    congestion_prob = model.predict(features)
    return congestion_prob

# Proactive resource allocation
if predict_congestion(network) > 0.7:
    reallocate_bandwidth()
```

### 2. **5G Network Slicing Integration**
Implement network slicing for isolated traffic classes:
```python
class NetworkSlice:
    def __init__(self, slice_id, guaranteed_bandwidth):
        self.slice_id = slice_id
        self.guaranteed_bandwidth = guaranteed_bandwidth
        self.traffic_types = []
    
    def allocate_resources(self):
        # Implement resource block allocation
        pass
```

### 3. **Cybersecurity Module**
Add intrusion detection and mitigation:
```python
def detect_anomaly(packet_stream):
    # Check for DoS attacks, man-in-the-middle
    if packet_rate > threshold:
        trigger_defense()
```

### 4. **Renewable Integration**
Model communication requirements for DERs:
```python
class DistributedEnergyResource:
    def __init__(self, der_id, generation_capacity):
        self.der_id = der_id
        self.generation_capacity = generation_capacity
    
    def send_telemetry(self):
        # Generate status packets
        pass
```

### 5. **Optimization Algorithms**
Implement advanced scheduling:
- **Genetic Algorithm** for bandwidth allocation
- **Deep Reinforcement Learning** for routing
- **Integer Linear Programming** for resource optimization


## License
MIT License

## Contact
Tahseen Zaman  
tahseenzaman11@gmail.com  
[GitHub](https://github.com/Tahseen-Zaman)

---