comparator = CrossEnvironmentComparator("quantum_results.csv", "classical_results.csv")
comparator.load_data()
results = comparator.generate_comparison_report()
