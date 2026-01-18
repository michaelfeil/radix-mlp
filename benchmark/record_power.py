#!/usr/bin/env python3
"""
CUDA Power Recording Script

Records real-time power draw and metrics from specified CUDA devices.
Supports concurrent recording with benchmark tests and generates CSV output for analysis.

Usage:
    python record_power.py --devices 1,2,3,5 --names config1,config2,config3,config4

Requirements:
    pip install pynvml pandas matplotlib
"""

import os
import sys
import time
import signal
import argparse
import threading
import csv
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. Install with: pip install pynvml")
    PYNVML_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available for plotting")
    PLOTTING_AVAILABLE = False


@dataclass
class PowerReading:
    """Single power reading from a CUDA device."""

    timestamp: datetime
    device_id: int
    device_name: str
    power_draw_w: float
    temperature_c: float
    memory_usage_gb: float
    memory_total_gb: float
    utilization_pct: float
    graphics_clock_mhz: float
    memory_clock_mhz: float


class PowerRecorder:
    """Main class for recording CUDA device power consumption."""

    def __init__(
        self,
        device_ids: List[int],
        device_names: List[str],
        output_file: str = "power_data.csv",
        sampling_rate: float = 10.0,
        verbose: bool = False,
    ):
        self.device_ids = device_ids
        self.device_names = device_names
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.is_recording = False
        self.readings: List[PowerReading] = []
        self.device_handles: Dict[int, any] = {}
        self.csv_file = None
        self.csv_writer = None
        self.recording_thread = None

        # Validate inputs
        if len(device_ids) != len(device_names):
            raise ValueError("Number of devices must match number of names")

        if not PYNVML_AVAILABLE:
            raise RuntimeError("pynvml is required but not available")

    def initialize_devices(self) -> bool:
        """Initialize pynvml and device handles."""
        try:
            pynvml.nvmlInit()
            if self.verbose:
                print(
                    f"Initialized NVML, driver version: {pynvml.nvmlSystemGetDriverVersion()}"
                )

            # Initialize device handles
            for device_id, device_name in zip(self.device_ids, self.device_names):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    self.device_handles[device_id] = handle

                    # Get device info for validation
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    if self.verbose:
                        print(f"Device {device_id}: {name} -> '{device_name}'")

                except pynvml.NVMLError as e:
                    print(f"Error initializing device {device_id}: {e}")
                    return False

            if self.verbose:
                print(f"Successfully initialized {len(self.device_handles)} devices")

            return True

        except pynvml.NVMLError as e:
            print(f"Error initializing NVML: {e}")
            return False

    def get_device_metrics(self, device_id: int) -> Optional[PowerReading]:
        """Get current power metrics from a device."""
        if device_id not in self.device_handles:
            return None

        handle = self.device_handles[device_id]

        try:
            # Power draw (in milliwatts, convert to watts)
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_draw_w = power_draw_mw / 1000.0

            # Temperature (in Celsius)
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temp_c = 0.0

            # Memory usage (in GB)
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage_gb = memory_info.used / (1024 * 1024 * 1024)
                memory_total_gb = memory_info.total / (1024 * 1024 * 1024)
            except pynvml.NVMLError:
                memory_usage_gb = 0.0
                memory_total_gb = 0.0

            # GPU utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_pct = utilization.gpu
            except pynvml.NVMLError:
                utilization_pct = 0.0

            # GPU clock frequencies
            try:
                graphics_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
            except pynvml.NVMLError:
                graphics_clock_mhz = 0.0

            try:
                memory_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
            except pynvml.NVMLError:
                memory_clock_mhz = 0.0

            return PowerReading(
                timestamp=datetime.now(),
                device_id=device_id,
                device_name=self.device_names[self.device_ids.index(device_id)],
                power_draw_w=power_draw_w,
                temperature_c=temp_c,
                memory_usage_gb=memory_usage_gb,
                memory_total_gb=memory_total_gb,
                utilization_pct=utilization_pct,
                graphics_clock_mhz=graphics_clock_mhz,
                memory_clock_mhz=memory_clock_mhz,
            )

        except pynvml.NVMLError as e:
            if self.verbose:
                print(f"Error reading metrics from device {device_id}: {e}")
            return None

    def initialize_csv(self):
        """Initialize CSV file for writing."""
        try:
            self.csv_file = open(self.output_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)

            # Write header
            header = [
                "timestamp",
                "device_id",
                "device_name",
                "power_draw_w",
                "temperature_c",
                "memory_usage_gb",
                "memory_total_gb",
                "utilization_pct",
                "graphics_clock_mhz",
                "memory_clock_mhz",
            ]
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            if self.verbose:
                print(f"Initialized CSV output: {self.output_file}")

        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            raise

    def recording_loop(self):
        """Main recording loop running in separate thread."""
        if self.verbose:
            print(f"Starting recording loop at {self.sampling_rate} Hz")

        sleep_interval = 1.0 / self.sampling_rate

        while self.is_recording:
            loop_start = time.time()

            # Collect readings from all devices
            for device_id in self.device_ids:
                reading = self.get_device_metrics(device_id)
                if reading:
                    self.readings.append(reading)

                    # Write to CSV immediately
                    if self.csv_writer:
                        row = [
                            reading.timestamp.isoformat(),
                            reading.device_id,
                            reading.device_name,
                            f"{reading.power_draw_w:.2f}",
                            f"{reading.temperature_c:.1f}",
                            f"{reading.memory_usage_gb:.2f}",
                            f"{reading.memory_total_gb:.1f}",
                            f"{reading.utilization_pct:.1f}",
                            f"{reading.graphics_clock_mhz:.0f}",
                            f"{reading.memory_clock_mhz:.0f}",
                        ]
                        self.csv_writer.writerow(row)

            # Flush CSV periodically
            if len(self.readings) % 10 == 0 and self.csv_file:
                self.csv_file.flush()

            # Calculate sleep time to maintain sampling rate
            loop_duration = time.time() - loop_start
            sleep_time = max(0, sleep_interval - loop_duration)
            time.sleep(sleep_time)

        if self.verbose:
            print("Recording loop stopped")

    def start_recording(self) -> bool:
        """Start power recording."""
        if self.is_recording:
            print("Recording already in progress")
            return False

        if not self.initialize_devices():
            return False

        try:
            self.initialize_csv()
            self.is_recording = True

            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self.recording_loop, daemon=True
            )
            self.recording_thread.start()

            print(
                f"Started recording {len(self.device_ids)} devices to {self.output_file}"
            )
            print(f"Sampling rate: {self.sampling_rate} Hz")
            print("Press Ctrl+C to stop recording")

            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def stop_recording(self):
        """Stop power recording and cleanup."""
        if not self.is_recording:
            return

        print("Stopping recording...")
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        # Cleanup NVML
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

        print(
            f"Recording stopped. {len(self.readings)} readings saved to {self.output_file}"
        )

    def generate_plots(self, output_dir: str = ".") -> bool:
        """Generate power consumption plots from recorded data."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib/seaborn not available for plotting")
            return False

        if not self.readings:
            print("No data to plot")
            return False

        try:
            # Convert readings to DataFrame
            data = []
            for reading in self.readings:
                data.append(
                    {
                        "timestamp": reading.timestamp,
                        "device_id": reading.device_id,
                        "device_name": reading.device_name,
                        "power_draw_w": reading.power_draw_w,
                        "temperature_c": reading.temperature_c,
                        "memory_usage_gb": reading.memory_usage_gb,
                        "memory_total_gb": reading.memory_total_gb,
                        "utilization_pct": reading.utilization_pct,
                        "graphics_clock_mhz": reading.graphics_clock_mhz,
                        "memory_clock_mhz": reading.memory_clock_mhz,
                    }
                )

            df = pd.DataFrame(data)

            # Convert to relative time from start
            start_time = df["timestamp"].min()
            df["relative_time"] = (df["timestamp"] - start_time).dt.total_seconds()

            # Create side-by-side plots: Power & Frequency over time
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
            sns.set_style("whitegrid")

            # Plot 1: Power over time (left)
            for device_name in df["device_name"].unique():
                device_data = df[df["device_name"] == device_name]
                ax1.plot(
                    device_data["relative_time"],
                    device_data["power_draw_w"],
                    label=device_name,
                    linewidth=2,
                )

            ax1.set_ylabel("Power Draw (W)", fontsize=12)
            ax1.set_title("Time vs Power Draw", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel("Time (seconds)", fontsize=12)

            # Plot 2: GPU Clock Frequency over time (right)
            for device_name in df["device_name"].unique():
                device_data = df[df["device_name"] == device_name]
                ax2.plot(
                    device_data["relative_time"],
                    device_data["graphics_clock_mhz"],
                    label=f"{device_name} (Graphics)",
                    linewidth=2,
                    linestyle="-",
                )
                ax2.plot(
                    device_data["relative_time"],
                    device_data["memory_clock_mhz"],
                    label=f"{device_name} (Memory)",
                    linewidth=2,
                    linestyle="--",
                )

            ax2.set_ylabel("Clock Frequency (MHz)", fontsize=12)
            ax2.set_title("Time vs Clock Frequency", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel("Time (seconds)", fontsize=12)

            # Adjust layout and save
            plt.subplots_adjust(wspace=0.3)
            plt.tight_layout()

            # Save dual plot
            plot_file = os.path.join(output_dir, "power_frequency.pdf")
            plt.savefig(plot_file, format="pdf", bbox_inches="tight")
            plt.close()

            print(f"Power & Frequency plot saved to {plot_file}")

            # Create additional detailed plots
            self.create_detailed_plots(df, output_dir)

            # Generate summary statistics
            self.generate_summary_stats(df, output_dir)

            return True

        except Exception as e:
            print(f"Error generating plots: {e}")
            return False

    def create_detailed_plots(self, df: pd.DataFrame, output_dir: str):
        """Create additional detailed plots for comprehensive analysis."""
        try:
            # Plot 1: Temperature, GPU Utilization and VRAM Usage
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            sns.set_style("whitegrid")

            for device_name in df["device_name"].unique():
                device_data = df[df["device_name"] == device_name]

                # Temperature plot
                ax1.plot(
                    device_data["relative_time"],
                    device_data["temperature_c"],
                    label=device_name,
                    linewidth=2,
                )

                # GPU Utilization plot
                ax2.plot(
                    device_data["relative_time"],
                    device_data["utilization_pct"],
                    label=device_name,
                    linewidth=2,
                )

                # VRAM Usage plot
                ax3.plot(
                    device_data["relative_time"],
                    device_data["memory_usage_gb"],
                    label=device_name,
                    linewidth=2,
                )
                # Add horizontal line for max VRAM
                max_vram = device_data["memory_total_gb"].iloc[0]

            ax1.set_ylabel("Temperature (°C)", fontsize=12)
            ax1.set_title("CUDA Device Temperature Over Time", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_ylabel("GPU Utilization (%)", fontsize=12)
            ax2.set_title("CUDA Device GPU Utilization Over Time", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3.set_ylabel("VRAM Usage (GB)", fontsize=12)
            ax3.set_title("CUDA Device VRAM Usage Over Time", fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.xlabel("Time (seconds)", fontsize=12)
            plt.tight_layout()

            temp_plot_file = os.path.join(output_dir, "temperature_utilization.pdf")
            plt.savefig(temp_plot_file, format="pdf", bbox_inches="tight")
            plt.close()

            # Plot 2: Power vs Clock Frequency and VRAM Usage Scatter Plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
            sns.set_style("whitegrid")

            for device_name in df["device_name"].unique():
                device_data = df[df["device_name"] == device_name]

                # Power vs Graphics Clock
                ax1.scatter(
                    device_data["graphics_clock_mhz"],
                    device_data["power_draw_w"],
                    label=device_name,
                    alpha=0.6,
                    s=20,
                )

                # Power vs Memory Clock
                ax2.scatter(
                    device_data["memory_clock_mhz"],
                    device_data["power_draw_w"],
                    label=device_name,
                    alpha=0.6,
                    s=20,
                )

                # Power vs GPU Utilization
                ax3.scatter(
                    device_data["utilization_pct"],
                    device_data["power_draw_w"],
                    label=device_name,
                    alpha=0.6,
                    s=20,
                )

                # Power vs VRAM Usage
                ax4.scatter(
                    device_data["memory_usage_gb"],
                    device_data["power_draw_w"],
                    label=device_name,
                    alpha=0.6,
                    s=20,
                )

            # Configure subplots
            ax1.set_xlabel("Graphics Clock (MHz)", fontsize=12)
            ax1.set_ylabel("Power Draw (W)", fontsize=12)
            ax1.set_title("Power vs Graphics Clock", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel("Memory Clock (MHz)", fontsize=12)
            ax2.set_ylabel("Power Draw (W)", fontsize=12)
            ax2.set_title("Power vs Memory Clock", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3.set_xlabel("GPU Utilization (%)", fontsize=12)
            ax3.set_ylabel("Power Draw (W)", fontsize=12)
            ax3.set_title("Power vs GPU Utilization", fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            ax4.set_xlabel("VRAM Usage (GB)", fontsize=12)
            ax4.set_ylabel("Power Draw (W)", fontsize=12)
            ax4.set_title("Power vs VRAM Usage", fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            scatter_plot_file = os.path.join(output_dir, "power_clock_scatter.pdf")
            plt.savefig(scatter_plot_file, format="pdf", bbox_inches="tight")
            plt.close()

            print(f"Detailed plots saved:")
            print(f"  - Temperature & Utilization: {temp_plot_file}")
            print(f"  - Power vs Metrics Scatter: {scatter_plot_file}")

        except Exception as e:
            print(f"Error creating detailed plots: {e}")

    def generate_summary_stats(self, df: pd.DataFrame, output_dir: str):
        """Generate summary statistics and save to file."""
        try:
            stats_file = os.path.join(output_dir, "power_summary.txt")

            with open(stats_file, "w") as f:
                f.write("CUDA Device Power Consumption Summary\n")
                f.write("=" * 50 + "\n\n")

                for device_name in df["device_name"].unique():
                    device_data = df[df["device_name"] == device_name]

                    f.write(f"Device: {device_name}\n")
                    f.write("-" * 30 + "\n")

                    # Power statistics
                    power_stats = device_data["power_draw_w"].describe()
                    f.write(f"Power Draw (W):\n")
                    f.write(f"  Mean: {power_stats['mean']:.2f}\n")
                    f.write(f"  Std:  {power_stats['std']:.2f}\n")
                    f.write(f"  Min:  {power_stats['min']:.2f}\n")
                    f.write(f"  Max:  {power_stats['max']:.2f}\n")
                    f.write(
                        f"  P50:  {device_data['power_draw_w'].quantile(0.5):.2f}\n"
                    )
                    f.write(
                        f"  P90:  {device_data['power_draw_w'].quantile(0.9):.2f}\n"
                    )
                    f.write(
                        f"  P95:  {device_data['power_draw_w'].quantile(0.95):.2f}\n"
                    )

                    # Temperature statistics
                    temp_stats = device_data["temperature_c"].describe()
                    f.write(f"Temperature (°C):\n")
                    f.write(f"  Mean: {temp_stats['mean']:.1f}\n")
                    f.write(f"  Max:  {temp_stats['max']:.1f}\n")

                    # GPU Utilization statistics
                    util_stats = device_data["utilization_pct"].describe()
                    f.write(f"GPU Utilization (%):\n")
                    f.write(f"  Mean: {util_stats['mean']:.1f}\n")
                    f.write(f"  Max:  {util_stats['max']:.1f}\n")

                    # VRAM Usage statistics
                    vram_stats = device_data["memory_usage_gb"].describe()
                    f.write(f"VRAM Usage (GB):\n")
                    f.write(f"  Mean: {vram_stats['mean']:.2f}\n")
                    f.write(f"  Max:  {vram_stats['max']:.2f}\n")

                    # Graphics Clock statistics
                    graphics_clock_stats = device_data["graphics_clock_mhz"].describe()
                    f.write(f"Graphics Clock (MHz):\n")
                    f.write(f"  Mean: {graphics_clock_stats['mean']:.0f}\n")
                    f.write(f"  Min:  {graphics_clock_stats['min']:.0f}\n")
                    f.write(f"  Max:  {graphics_clock_stats['max']:.0f}\n")

                    # Memory Clock statistics
                    memory_clock_stats = device_data["memory_clock_mhz"].describe()
                    f.write(f"Memory Clock (MHz):\n")
                    f.write(f"  Mean: {memory_clock_stats['mean']:.0f}\n")
                    f.write(f"  Min:  {memory_clock_stats['min']:.0f}\n")
                    f.write(f"  Max:  {memory_clock_stats['max']:.0f}\n")

                    f.write(f"  Total readings: {len(device_data)}\n")
                    f.write(
                        f"  Recording duration: {(device_data['timestamp'].max() - device_data['timestamp'].min()).total_seconds():.1f}s\n\n"
                    )

            print(f"Summary statistics saved to {stats_file}")

        except Exception as e:
            print(f"Error generating summary stats: {e}")


# Global recorder instance for signal handling
recorder: Optional[PowerRecorder] = None


def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals for graceful shutdown."""
    global recorder
    if recorder:
        print("\nReceived signal to stop recording...")
        recorder.stop_recording()
    sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Record CUDA device power draw and metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python record_power.py --devices 1,2,3,5 --names config1,config2,config3,config4
  
  # With custom settings
  python record_power.py --devices 0,1 --names gpu0,gpu1 --rate 5.0 --output power.csv
  
  # Time-limited recording
  python record_power.py --devices 0 --name test_gpu --duration 60
        """,
    )

    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated list of CUDA device IDs (e.g., 1,2,3,5)",
    )

    parser.add_argument(
        "--names",
        required=True,
        help="Comma-separated list of device names (must match number of devices)",
    )

    parser.add_argument(
        "--output",
        default="power_data.csv",
        help="Output CSV file (default: power_data.csv)",
    )

    parser.add_argument(
        "--rate", type=float, default=5.0, help="Sampling rate in Hz (default: 5.0)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        help="Recording duration in seconds (default: run until manually stopped)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate power consumption plots after recording",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Parse device IDs and names
    try:
        device_ids = [int(d.strip()) for d in args.devices.split(",")]
        device_names = [n.strip() for n in args.names.split(",")]
    except ValueError as e:
        print(f"Error parsing devices/names: {e}")
        return 1

    # Validate inputs
    if len(device_ids) != len(device_names):
        print("Error: Number of devices must match number of names")
        return 1

    if args.rate <= 0 or args.rate > 100:
        print("Error: Sampling rate must be between 0 and 100 Hz")
        return 1

    # Create recorder instance
    global recorder
    try:
        recorder = PowerRecorder(
            device_ids=device_ids,
            device_names=device_names,
            output_file=args.output,
            sampling_rate=args.rate,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error creating recorder: {e}")
        return 1

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start recording
    if not recorder.start_recording():
        return 1

    try:
        # Run for specified duration or until interrupted
        if args.duration:
            print(f"Recording for {args.duration} seconds...")
            time.sleep(args.duration)
            recorder.stop_recording()
        else:
            print("Recording... Press Ctrl+C to stop")
            # Keep main thread alive
            while recorder.is_recording:
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass  # Handled by signal handler

    # Generate plots if requested
    if args.plot:
        output_dir = os.path.dirname(args.output) or "."
        recorder.generate_plots(output_dir)

    print("Recording completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
