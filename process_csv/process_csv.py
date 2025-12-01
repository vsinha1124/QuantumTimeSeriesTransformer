# normalize_throughput.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

class ThroughputNormalizer:
    """Normalize throughput calculations from benchmark CSV files"""

    # Constants from benchmark configurations
    QUANTUM_CONSTANTS = {
        'shots_per_run': 1000,
        'tokens_per_circuit': 2,  # All quantum mini circuits process 2 tokens
    }

    CLASSICAL_CONSTANTS = {
        'batch_size': 32,
        'seq_len': 2,  # Matches quantum's 2 tokens
        'embed_dim': 64,  # Standard embedding dimension
    }

    # Architecture-specific mappings
    ARCHITECTURE_INFO = {
        # Quantum architectures
        'Quantum_LCU_Linear': {
            'type': 'quantum',
            'qubits': 3,
            'description': 'Linear combination of token unitaries (basic mixing)',
            'classical_equivalent': 'Classical_LCU_Equivalent'
        },
        'Quantum_QSVT_Single': {
            'type': 'quantum',
            'qubits': 3,
            'description': 'Nonlinear transform on single unitary (feature transformation)',
            'classical_equivalent': 'Classical_QSVT_Single_Equivalent'
        },
        'Quantum_QSVT_Full': {
            'type': 'quantum',
            'qubits': 4,
            'description': 'Full nonlinear transform on mixed unitaries (complete attention)',
            'classical_equivalent': 'Classical_QSVT_Full_Equivalent'
        },

        # Classical architectures
        'Classical_LCU_Equivalent': {
            'type': 'classical',
            'description': 'Linear token mixing (equivalent to Quantum LCU)',
            'quantum_equivalent': 'Quantum_LCU_Linear'
        },
        'Classical_QSVT_Single_Equivalent': {
            'type': 'classical',
            'description': 'Nonlinear feature transform (equivalent to Quantum QSVT Single)',
            'quantum_equivalent': 'Quantum_QSVT_Single'
        },
        'Classical_QSVT_Full_Equivalent': {
            'type': 'classical',
            'description': 'Complete attention mechanism (equivalent to Quantum QSVT Full)',
            'quantum_equivalent': 'Quantum_QSVT_Full'
        }
    }

    def __init__(self, target_precision=0.05, parallel_efficiency=0.8):
        """
        Args:
            target_precision: Desired statistical precision for quantum (ε)
            parallel_efficiency: Efficiency factor for classical parallel processing
        """
        self.target_precision = target_precision
        self.parallel_efficiency = parallel_efficiency
        self.shots_for_precision = int(1 / (target_precision ** 2))

    def quantum_to_classical_dimensions(self, qubits):
        """
        Convert qubit count to equivalent classical embedding dimensions.

        Based on information theory and empirical mapping from Quixer paper:
        - 3 qubits ≈ 64 dimensions (2³ = 8 complex states → scaled)
        - 4 qubits ≈ 128 dimensions (2⁴ = 16 complex states → scaled)
        - 6 qubits ≈ 512 dimensions (2⁶ = 64 complex states → scaled)

        General approximation: 8 × 2^qubits
        """
        return 8 * (2 ** qubits)

    def normalize_quantum_throughput(self, row):
        """
        Calculate normalized throughput for quantum benchmark results.

        Args:
            row: DataFrame row with quantum benchmark data

        Returns:
            Dictionary with normalized throughput metrics
        """
        # Extract raw metrics from CSV
        raw_throughput = row.get('throughput_samples_s', 0)  # circuits/second (old calculation)
        execution_time = row.get('execution_time_s', 0)
        success_prob = row.get('success_probability', 1.0)
        qubits = row.get('qubits', 0)

        # Get architecture info
        arch_name = row.get('architecture', '')
        arch_info = self.ARCHITECTURE_INFO.get(arch_name, {})

        # Reconstruct shots from old calculation
        # old_throughput = (shots * success_prob) / execution_time
        # So: shots = (old_throughput * execution_time) / success_prob
        if execution_time > 0 and success_prob > 0:
            shots = (raw_throughput * execution_time) / success_prob
        else:
            shots = self.QUANTUM_CONSTANTS['shots_per_run']

        # Calculate normalized metrics
        tokens_per_circuit = self.QUANTUM_CONSTANTS['tokens_per_circuit']
        dims_per_quantum_token = self.quantum_to_classical_dimensions(qubits)

        # 1. Raw quantum throughput (circuits/second)
        raw_circuits_per_second = raw_throughput

        # 2. Quantum tokens per second
        quantum_tokens_per_second = raw_circuits_per_second * tokens_per_circuit

        # 3. Normalized throughput (without statistical adjustment)
        # dimension-tokens/second = circuits/sec × tokens/circuit × dimensions/token
        norm_throughput = raw_circuits_per_second * tokens_per_circuit * dims_per_quantum_token

        # 4. Adjusted for statistical requirements
        # Divide by shots needed for target precision
        adj_throughput = norm_throughput / self.shots_for_precision

        return {
            'architecture': arch_name,
            'description': arch_info.get('description', ''),
            'original_raw_throughput': raw_throughput,
            'raw_circuits_per_second': raw_circuits_per_second,
            'quantum_tokens_per_second': quantum_tokens_per_second,
            'qubits': qubits,
            'dims_per_quantum_token': dims_per_quantum_token,
            'norm_quantum_throughput': norm_throughput,
            'shots_for_precision': self.shots_for_precision,
            'adj_quantum_throughput': adj_throughput,
            'type': 'quantum',
            'device_used': row.get('device_used', 'unknown'),
            'timestamp': row.get('timestamp', '')
        }

    def normalize_classical_throughput(self, row):
        """
        Calculate normalized throughput for classical benchmark results.

        Args:
            row: DataFrame row with classical benchmark data

        Returns:
            Dictionary with normalized throughput metrics
        """
        # Extract raw metrics from CSV
        raw_throughput = row.get('throughput_samples_s', 0)  # batches/second (old calculation)
        execution_time = row.get('execution_time_s', 0)

        # Get architecture info
        arch_name = row.get('architecture', '')
        arch_info = self.ARCHITECTURE_INFO.get(arch_name, {})

        # Calculate normalized metrics
        batch_size = self.CLASSICAL_CONSTANTS['batch_size']
        seq_len = self.CLASSICAL_CONSTANTS['seq_len']
        embed_dim = self.CLASSICAL_CONSTANTS['embed_dim']

        # Old calculation: throughput_samples_s = batch_size / execution_time
        # So raw_throughput is batches per second
        raw_batches_per_second = raw_throughput

        # Classical tokens per second
        classical_tokens_per_second = raw_batches_per_second * batch_size * seq_len

        # Normalized throughput (dimension-tokens/second)
        norm_throughput = (raw_batches_per_second * batch_size * seq_len *
                          embed_dim * self.parallel_efficiency)

        return {
            'architecture': arch_name,
            'description': arch_info.get('description', ''),
            'original_raw_throughput': raw_throughput,
            'raw_batches_per_second': raw_batches_per_second,
            'classical_tokens_per_second': classical_tokens_per_second,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'dims_per_classical_token': embed_dim,
            'norm_classical_throughput': norm_throughput,
            'parallel_efficiency': self.parallel_efficiency,
            'type': 'classical',
            'device_used': 'cpu/gpu',
            'timestamp': row.get('timestamp', '')
        }

    def create_comparison_table(self, quantum_df, classical_df):
        """
        Create fair comparison between quantum and classical architectures.

        Args:
            quantum_df: DataFrame with normalized quantum results
            classical_df: DataFrame with normalized classical results

        Returns:
            DataFrame with paired comparisons
        """
        comparison_rows = []

        # Map quantum to classical architectures
        for q_name in ['Quantum_LCU_Linear', 'Quantum_QSVT_Single', 'Quantum_QSVT_Full']:
            c_name = self.ARCHITECTURE_INFO[q_name].get('classical_equivalent')

            # Get latest results for each architecture (take mean if multiple runs)
            q_results = quantum_df[quantum_df['architecture'] == q_name]
            c_results = classical_df[classical_df['architecture'] == c_name]

            if len(q_results) > 0 and len(c_results) > 0:
                # Use average of all runs
                q_avg = q_results.mean(numeric_only=True)
                c_avg = c_results.mean(numeric_only=True)

                # Get descriptions
                q_desc = q_results.iloc[0]['description'] if 'description' in q_results.columns else ''
                c_desc = c_results.iloc[0]['description'] if 'description' in c_results.columns else ''

                # Calculate comparison ratios
                # 1. Raw ratio (old unfair comparison)
                # Quantum: circuits/second, Classical: batches/second
                # Need to convert to common units first
                q_raw_tokens_sec = q_avg.get('quantum_tokens_per_second', 0)
                c_raw_tokens_sec = c_avg.get('classical_tokens_per_second', 0)

                if q_raw_tokens_sec > 0:
                    raw_ratio = c_raw_tokens_sec / q_raw_tokens_sec
                else:
                    raw_ratio = float('inf')

                # 2. Normalized ratio (dimension-tokens/second)
                q_norm = q_avg.get('norm_quantum_throughput', 0)
                c_norm = c_avg.get('norm_classical_throughput', 0)

                if q_norm > 0:
                    norm_ratio = c_norm / q_norm
                else:
                    norm_ratio = float('inf')

                # 3. Adjusted ratio (accounting for quantum statistics)
                q_adj = q_avg.get('adj_quantum_throughput', 0)

                if q_adj > 0:
                    adj_ratio = c_norm / q_adj
                else:
                    adj_ratio = float('inf')

                comparison_rows.append({
                    'quantum_architecture': q_name,
                    'classical_architecture': c_name,
                    'quantum_description': q_desc,
                    'classical_description': c_desc,

                    # Raw metrics (old calculations)
                    'q_raw_circuits_per_second': q_avg.get('raw_circuits_per_second', 0),
                    'c_raw_batches_per_second': c_avg.get('raw_batches_per_second', 0),
                    'q_tokens_per_second': q_raw_tokens_sec,
                    'c_tokens_per_second': c_raw_tokens_sec,
                    'raw_ratio_c_over_q': raw_ratio,

                    # Normalized metrics
                    'q_norm_throughput': q_norm,
                    'c_norm_throughput': c_norm,
                    'norm_ratio_c_over_q': norm_ratio,

                    # Adjusted metrics (most fair)
                    'q_adj_throughput': q_adj,
                    'adj_ratio_c_over_q': adj_ratio,

                    # Architecture details
                    'quantum_qubits': q_avg.get('qubits', 0),
                    'quantum_dims_per_token': q_avg.get('dims_per_quantum_token', 0),
                    'classical_embed_dim': c_avg.get('embed_dim', 0),
                    'quantum_device': q_results.iloc[0].get('device_used', 'unknown'),

                    # Statistical info
                    'shots_for_precision': q_avg.get('shots_for_precision', 0),
                    'parallel_efficiency': c_avg.get('parallel_efficiency', 0),
                    'target_precision': self.target_precision
                })

        return pd.DataFrame(comparison_rows)

    def process_csv_files(self, quantum_csv_paths, classical_csv_paths, output_dir='normalized_results'):
        """
        Process multiple CSV files and generate normalized results.

        Args:
            quantum_csv_paths: List of paths to quantum benchmark CSVs
            classical_csv_paths: List of paths to classical benchmark CSVs
            output_dir: Directory to save output files

        Returns:
            Tuple of (quantum_results_df, classical_results_df, comparison_df)
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Load and process quantum CSVs
        quantum_dfs = []
        for csv_path in quantum_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded quantum CSV: {csv_path} ({len(df)} rows)")

                # Add normalized metrics
                normalized_rows = []
                for _, row in df.iterrows():
                    normalized = self.normalize_quantum_throughput(row)
                    normalized_rows.append(normalized)

                quantum_dfs.append(pd.DataFrame(normalized_rows))
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

        # Load and process classical CSVs
        classical_dfs = []
        for csv_path in classical_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded classical CSV: {csv_path} ({len(df)} rows)")

                # Add normalized metrics
                normalized_rows = []
                for _, row in df.iterrows():
                    normalized = self.normalize_classical_throughput(row)
                    normalized_rows.append(normalized)

                classical_dfs.append(pd.DataFrame(normalized_rows))
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

        # Combine all results
        if quantum_dfs:
            quantum_results = pd.concat(quantum_dfs, ignore_index=True)
            quantum_output = Path(output_dir) / f'normalized_quantum_{timestamp}.csv'
            quantum_results.to_csv(quantum_output, index=False)
            print(f"Saved normalized quantum results to: {quantum_output}")
        else:
            quantum_results = pd.DataFrame()
            print("No quantum results to process")

        if classical_dfs:
            classical_results = pd.concat(classical_dfs, ignore_index=True)
            classical_output = Path(output_dir) / f'normalized_classical_{timestamp}.csv'
            classical_results.to_csv(classical_output, index=False)
            print(f"Saved normalized classical results to: {classical_output}")
        else:
            classical_results = pd.DataFrame()
            print("No classical results to process")

        # Create comparison table if we have both quantum and classical results
        if not quantum_results.empty and not classical_results.empty:
            comparison_df = self.create_comparison_table(quantum_results, classical_results)
            comparison_output = Path(output_dir) / f'fair_comparison_{timestamp}.csv'
            comparison_df.to_csv(comparison_output, index=False)
            print(f"Saved fair comparison to: {comparison_output}")

            # Generate summary report
            self.generate_summary_report(comparison_df, output_dir, timestamp)
        else:
            comparison_df = pd.DataFrame()
            print("Cannot create comparison - missing quantum or classical results")

        return quantum_results, classical_results, comparison_df

    def generate_summary_report(self, comparison_df, output_dir, timestamp):
        """Generate a human-readable summary report"""
        report_path = Path(output_dir) / f'summary_report_{timestamp}.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FAIR THROUGHPUT COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target precision (quantum): {self.target_precision}\n")
            f.write(f"Parallel efficiency (classical): {self.parallel_efficiency}\n")
            f.write(f"Shots required for precision: {self.shots_for_precision:,}\n\n")

            f.write("SUMMARY OF FINDINGS\n")
            f.write("-" * 80 + "\n\n")

            for _, row in comparison_df.iterrows():
                f.write(f"{row['quantum_architecture']} ↔ {row['classical_architecture']}\n")
                f.write(f"  Raw ratio (C/Q): {row['raw_ratio_c_over_q']:.1f}x\n")
                f.write(f"  Normalized ratio: {row['norm_ratio_c_over_q']:.1f}x\n")
                f.write(f"  Adjusted ratio: {row['adj_ratio_c_over_q']:.1f}x\n")

                # Interpretation
                adj_ratio = row['adj_ratio_c_over_q']
                if adj_ratio > 1000:
                    conclusion = "Classical is orders of magnitude faster"
                elif adj_ratio > 100:
                    conclusion = "Classical is significantly faster"
                elif adj_ratio > 10:
                    conclusion = "Classical is moderately faster"
                elif adj_ratio > 1:
                    conclusion = "Classical is slightly faster"
                elif adj_ratio < 1:
                    conclusion = "QUANTUM IS FASTER (theoretical breakthrough!)"
                else:
                    conclusion = "Performance is roughly equal"

                f.write(f"  Conclusion: {conclusion}\n\n")

            # Overall statistics
            if len(comparison_df) > 0:
                avg_raw_ratio = comparison_df['raw_ratio_c_over_q'].mean()
                avg_norm_ratio = comparison_df['norm_ratio_c_over_q'].mean()
                avg_adj_ratio = comparison_df['adj_ratio_c_over_q'].mean()

                f.write("OVERALL STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average raw ratio (C/Q): {avg_raw_ratio:.1f}x\n")
                f.write(f"Average normalized ratio: {avg_norm_ratio:.1f}x\n")
                f.write(f"Average adjusted ratio: {avg_adj_ratio:.1f}x\n\n")

                # Quantum scaling insight
                f.write("QUANTUM SCALING INSIGHTS\n")
                f.write("-" * 80 + "\n")
                for qubits in [3, 4, 6]:
                    quantum_rows = comparison_df[comparison_df['quantum_qubits'] == qubits]
                    if len(quantum_rows) > 0:
                        avg_ratio = quantum_rows['adj_ratio_c_over_q'].mean()
                        f.write(f"{qubits} qubits: Classical is {avg_ratio:.1f}x faster after adjustments\n")

                f.write("\nNote: These results reflect current NISQ-era quantum hardware.\n")
                f.write("Quantum advantage may emerge at larger scales (>50 qubits).\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Normalize throughput calculations from benchmark CSV files')
    parser.add_argument('--quantum-csv', nargs='+', help='Path(s) to quantum benchmark CSV files')
    parser.add_argument('--classical-csv', nargs='+', help='Path(s) to classical benchmark CSV files')
    parser.add_argument('--output-dir', default='normalized_results', help='Output directory for normalized results')
    parser.add_argument('--precision', type=float, default=0.05, help='Target precision for quantum measurements (ε)')
    parser.add_argument('--parallel-eff', type=float, default=0.8, help='Parallel efficiency factor for classical')

    args = parser.parse_args()

    # If no files specified, use default names
    if not args.quantum_csv:
        args.quantum_csv = ['quantum_results.csv']
    if not args.classical_csv:
        args.classical_csv = ['classical_results.csv']

    # Create normalizer
    normalizer = ThroughputNormalizer(
        target_precision=args.precision,
        parallel_efficiency=args.parallel_eff
    )

    # Process files
    quantum_results, classical_results, comparison = normalizer.process_csv_files(
        args.quantum_csv,
        args.classical_csv,
        args.output_dir
    )

    # Print quick summary to console
    if not comparison.empty:
        print("\n" + "=" * 80)
        print("QUICK SUMMARY")
        print("=" * 80)

        for _, row in comparison.iterrows():
            print(f"\n{row['quantum_architecture']} ↔ {row['classical_architecture']}")
            print(f"  Adjusted ratio: {row['adj_ratio_c_over_q']:.1f}x (Classical/Quantum)")

            if row['adj_ratio_c_over_q'] < 1:
                print(f"  ⚡ QUANTUM ADVANTAGE DETECTED!")
            elif row['adj_ratio_c_over_q'] < 10:
                print(f"  ⚖️ Competitive performance")
            else:
                print(f"  🐌 Classical significantly faster")

if __name__ == "__main__":
    main()
