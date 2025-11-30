# cross_environment_comparison.py
import pandas as pd
import numpy as np
from datetime import datetime

class CrossEnvironmentComparator:
    """Compare quantum and classical results from CSV files across different environments"""

    def __init__(self, quantum_csv="quantum_results.csv", classical_csv="classical_results.csv"):
        self.quantum_csv = quantum_csv
        self.classical_csv = classical_csv
        self.quantum_data = None
        self.classical_data = None

    def load_data(self):
        """Load data from CSV files"""
        try:
            self.quantum_data = pd.read_csv(self.quantum_csv)
            self.classical_data = pd.read_csv(self.classical_csv)
            print(f"Loaded {len(self.quantum_data)} quantum results from {self.quantum_csv}")
            print(f"Loaded {len(self.classical_data)} classical results from {self.classical_csv}")
            return True
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            return False

    def map_equivalent_architectures(self):
        """Define mapping between quantum and classical architectures"""
        return {
            'Quantum_LCU_Linear': 'Classical_LCU_Equivalent',
            'Quantum_QSVT_Single': 'Classical_QSVT_Single_Equivalent',
            'Quantum_QSVT_Full': 'Classical_QSVT_Full_Equivalent'
        }

    def create_comparison_table(self):
        """Create direct comparison table from CSV data"""
        if self.quantum_data is None or self.classical_data is None:
            print("Please load data first using load_data()")
            return None

        mapping = self.map_equivalent_architectures()
        comparison_rows = []

        for quantum_arch, classical_arch in mapping.items():
            quantum_row = self.quantum_data[self.quantum_data['architecture'] == quantum_arch]
            classical_row = self.classical_data[self.classical_data['architecture'] == classical_arch]

            if len(quantum_row) > 0 and len(classical_row) > 0:
                q = quantum_row.iloc[0]
                c = classical_row.iloc[0]

                # Directly comparable metrics
                time_ratio = q['execution_time_s'] / c['execution_time_s']
                throughput_ratio = c['throughput_samples_s'] / q['throughput_samples_s']

                comparison_rows.append({
                    'quantum_architecture': quantum_arch,
                    'classical_architecture': classical_arch,
                    'quantum_time_s': q['execution_time_s'],
                    'classical_time_s': c['execution_time_s'],
                    'time_ratio_quantum_classical': time_ratio,
                    'quantum_throughput_samples_s': q['throughput_samples_s'],
                    'classical_throughput_samples_s': c['throughput_samples_s'],
                    'throughput_ratio_classical_quantum': throughput_ratio,
                    'quantum_qubits': q['qubits'],
                    'quantum_total_gates': q['total_gates'],
                    'classical_parameters': c['parameters'],
                    'quantum_environment': q['environment'],
                    'classical_environment': c['environment'],
                    'description': q['description']
                })

        return pd.DataFrame(comparison_rows)

    def generate_comparison_report(self, output_csv="comparison_report.csv"):
        """Generate comprehensive comparison report"""
        comparison_df = self.create_comparison_table()

        if comparison_df is None or len(comparison_df) == 0:
            print("No comparable architectures found")
            return

        print("CROSS-ENVIRONMENT ARCHITECTURE COMPARISON")
        print("=" * 120)
        print(f"{'Quantum Arch':<20} {'Classical Arch':<25} {'Q Time (s)':<10} {'C Time (s)':<10} {'Time Ratio':<10} {'Q Throughput':<12} {'C Throughput':<12} {'Tput Ratio':<10}")
        print("-" * 120)

        for _, row in comparison_df.iterrows():
            print(f"{row['quantum_architecture']:<20} {row['classical_architecture']:<25} {row['quantum_time_s']:<10.3f} {row['classical_time_s']:<10.6f} {row['time_ratio_quantum_classical']:<10.1f} {row['quantum_throughput_samples_s']:<12.0f} {row['classical_throughput_samples_s']:<12.0f} {row['throughput_ratio_classical_quantum']:<10.1f}")

        # Summary statistics
        print("\n" + "=" * 120)
        print("COMPARISON SUMMARY")
        print("=" * 120)

        avg_time_ratio = comparison_df['time_ratio_quantum_classical'].mean()
        avg_throughput_ratio = comparison_df['throughput_ratio_classical_quantum'].mean()

        print(f"Average Time Ratio (Quantum/Classical): {avg_time_ratio:.1f}x")
        print(f"Average Throughput Ratio (Classical/Quantum): {avg_throughput_ratio:.1f}x")
        print(f"Most Similar Performance: {comparison_df.loc[comparison_df['time_ratio_quantum_classical'].idxmin(), 'quantum_architecture']}")
        print(f"Largest Performance Gap: {comparison_df.loc[comparison_df['time_ratio_quantum_classical'].idxmax(), 'quantum_architecture']}")

        # Save detailed comparison to CSV
        comparison_df.to_csv(output_csv, index=False)
        print(f"\nDetailed comparison saved to {output_csv}")

        return comparison_df

    def performance_analysis(self):
        """Analyze performance characteristics"""
        comparison_df = self.create_comparison_table()
        if comparison_df is None:
            return

        print("\nPERFORMANCE ANALYSIS")
        print("=" * 80)

        # Scaling analysis
        quantum_scaling = comparison_df[['quantum_architecture', 'quantum_qubits', 'quantum_time_s']].copy()
        quantum_scaling['quantum_states'] = 2 ** quantum_scaling['quantum_qubits']

        classical_scaling = comparison_df[['classical_architecture', 'classical_parameters', 'classical_time_s']].copy()

        print("Quantum Scaling (Qubits vs Time):")
        for _, row in quantum_scaling.iterrows():
            print(f"  {row['quantum_architecture']}: {row['quantum_qubits']} qubits → {row['quantum_time_s']:.3f}s "
                  f"(States: {row['quantum_states']})")

        print("\nClassical Scaling (Parameters vs Time):")
        for _, row in classical_scaling.iterrows():
            print(f"  {row['classical_architecture']}: {row['classical_parameters']:,} params → {row['classical_time_s']:.6f}s")

# Usage example
if __name__ == "__main__":
    # This would be run after both benchmarks have generated their CSV files
    comparator = CrossEnvironmentComparator(
        quantum_csv="quantum_results.csv",
        classical_csv="classical_results.csv"
    )

    if comparator.load_data():
        comparison_results = comparator.generate_comparison_report()
        comparator.performance_analysis()
