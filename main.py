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