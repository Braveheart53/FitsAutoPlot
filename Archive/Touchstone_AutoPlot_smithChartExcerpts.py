#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMITH CHART METHODS - CORRECTED VERSION

This file contains the CORRECTED methods for SmithChartPlottermpld3 and TouchstonePlotCanvas
that properly use chart_type='z' or 'y' instead of the non-existent draw_admittance parameter.

Replace these methods in your Touchstone_AutoPlot.py with these corrected versions.

Key Changes:
- Removed invalid draw_admittance parameter from plot_s_smith() calls
- Use chart_type='z' for impedance and chart_type='y' for admittance
- Properly pass user-selected chart type from GUI to plotting methods
- Corrected SmithChartPlottermpld3.create_smith_chart()
- Corrected SmithChartPlottermpld3.export_smith_chart()
- Corrected TouchstonePlotCanvas.plot_smith_chart()
"""

# ============================================================================
# METHOD 1: SmithChartPlottermpld3.create_smith_chart() - CORRECTED
# ============================================================================

def create_smith_chart(self, network: Network, param_name: str = "S11",
                      chart_type: str = "z") -> Tuple[Figure, Network]:
    """Create an interactive Smith Chart visualization using scikit-rf native implementation.

    Parameters
    ----------
    network : Network
        Scikit-rf Network object containing single S-parameter data (1x1 network).
    param_name : str
        Parameter name for labeling (e.g., "S11", "S21").
    chart_type : str
        Chart type: "z" for impedance, "y" for admittance.

    Returns
    -------
    Tuple[Figure, Network]
        (matplotlib figure with proper Smith chart, network object).
    """
    try:
        # Create figure with proper size for Smith chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Use scikit-rf's native Smith chart plotting with all axes intact
        # CORRECTED: Removed invalid draw_admittance parameter
        network.plot_s_smith(
            ax=ax,
            chart_type=chart_type,  # 'z' for impedance, 'y' for admittance
            draw_labels=True,
            draw_vswr=True,
            show_legend=True
        )

        # Customize title
        chart_label = "Impedance" if chart_type == "z" else "Admittance"
        ax.set_title(f'{param_name} - Smith Chart ({chart_label})',
                    fontsize=14, fontweight='bold', pad=20)

        # Add mpld3 hover tooltips if available
        if MPLD3_AVAILABLE:
            try:
                # Get frequency and S-parameter data for tooltips
                freq_ghz = network.frequency.f / 1e9
                s_data = network.s[:, 0, 0]

                # Create tooltip text for each frequency point
                labels = []
                for i in range(len(freq_ghz)):
                    freq = freq_ghz[i]
                    s_val = s_data[i]
                    mag = np.abs(s_val)
                    phase = np.angle(s_val, deg=True)

                    # Calculate VSWR from reflection coefficient
                    if mag < 1.0:
                        vswr = (1 + mag) / (1 - mag + 1e-12)
                    else:
                        vswr = 10.0

                    label = (f"{param_name}<br>"
                            f"Frequency: {freq:.2f} GHz<br>"
                            f"Magnitude: {mag:.4f}<br>"
                            f"Phase: {phase:.1f}°<br>"
                            f"VSWR: {vswr:.2f}")
                    labels.append(label)

                # Attach tooltips to the plotted line/scatter
                for artist in ax.get_children():
                    if hasattr(artist, 'get_offsets') and len(artist.get_offsets()) > 0:
                        # Found scatter points, add tooltip
                        tooltip = plugins.PointHTMLTooltip(artist, labels, voffset=10, hoffset=10)
                        mpld3.plugins.connect(fig, tooltip)
                        break
                    elif hasattr(artist, 'get_xydata') and len(artist.get_xydata()) > 0:
                        # Found line, add tooltip at line points
                        xy_data = artist.get_xydata()
                        if len(xy_data) == len(labels):
                            tooltip = plugins.LineLabelTooltip(artist, labels)
                            mpld3.plugins.connect(fig, tooltip)
                            break

            except Exception as e:
                print(f"Warning: Could not add mpld3 tooltips: {e}")

        self.plots[param_name] = fig
        return fig, network

    except Exception as e:
        print(f"Error creating Smith chart: {e}")
        raise

# ============================================================================
# METHOD 2: SmithChartPlottermpld3.export_smith_chart() - CORRECTED
# ============================================================================

def export_smith_chart(self, network: Network, param_name: str, chart_type: str,
                      output_path: str, export_format: str) -> bool:
    """Export a Smith chart directly to file in the specified format.

    This method creates a fresh Smith chart and exports it, ensuring proper rendering.

    Parameters
    ----------
    network : Network
        Single S-parameter network (1x1).
    param_name : str
        Parameter name (e.g., "S11").
    chart_type : str
        "z" for impedance, "y" for admittance.
    output_path : str
        Output file path.
    export_format : str
        Export format: "html", "png", "svg", or "pdf".

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Create fresh figure with Smith chart
        fig, _ = self.create_smith_chart(network, param_name, chart_type)

        # Export based on format
        if export_format.lower() == "html":
            if not MPLD3_AVAILABLE:
                print("Warning: mpld3 not available, saving as PNG instead")
                fig.savefig(output_path.replace('.html', '.png'),
                           format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                return True

            html_str = mpld3.fig_to_html(fig)
            with open(output_path, 'w') as f:
                f.write(html_str)

        elif export_format.lower() == "png":
            fig.savefig(output_path, format='png', dpi=150, bbox_inches='tight')

        elif export_format.lower() == "svg":
            fig.savefig(output_path, format='svg', bbox_inches='tight')

        elif export_format.lower() == "pdf":
            fig.savefig(output_path, format='pdf', bbox_inches='tight')

        else:
            print(f"Unknown export format: {export_format}")
            plt.close(fig)
            return False

        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error exporting Smith chart: {e}")
        return False

# ============================================================================
# METHOD 3: TouchstonePlotCanvas.plot_smith_chart() - CORRECTED
# ============================================================================

def plot_smith_chart(self, network, title="Smith Chart", chart_type="z",
                    draw_labels=True, draw_vswr=True):
    """Plot Smith chart on the canvas using scikit-rf native implementation.
    
    Parameters
    ----------
    network : Network
        Single S-parameter network (1x1).
    title : str
        Plot title.
    chart_type : str
        "z" for impedance, "y" for admittance.
    draw_labels : bool
        Whether to draw Smith chart labels.
    draw_vswr : bool
        Whether to draw VSWR circles.
    """
    self.fig.clear()
    ax = self.fig.add_subplot(111)

    try:
        # Use scikit-rf's built-in Smith chart plotting with full axes
        # CORRECTED: Removed invalid draw_admittance parameter
        network.plot_s_smith(
            ax=ax,
            chart_type=chart_type,  # 'z' for impedance, 'y' for admittance
            draw_labels=draw_labels,
            draw_vswr=draw_vswr,
            show_legend=True
        )

        chart_label = "Impedance" if chart_type == "z" else "Admittance"
        ax.set_title(f"{title} - Smith Chart ({chart_label})", fontweight='bold', fontsize=12)

    except Exception as e:
        print(f"Smith chart plotting error: {e}")

        # Fallback to basic complex plane plot
        for i in range(network.nports):
            for j in range(network.nports):
                s_param = network.s[:, i, j]
                ax.plot(s_param.real, s_param.imag, label=f"S{i + 1}{j + 1}",
                       marker="o", markersize=3)

        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title(f"Complex Plane - {title}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis("equal")

    self.draw()

# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================
"""
BEFORE (INCORRECT):
    network.plot_s_smith(
        ax=ax,
        chart_type=chart_type,
        draw_labels=True,
        draw_vswr=True,
        draw_admittance=False,  ❌ INVALID PARAMETER - REMOVED
        show_legend=True,
        legend_location='upper right'  ❌ INVALID PARAMETER - REMOVED
    )

AFTER (CORRECT):
    network.plot_s_smith(
        ax=ax,
        chart_type=chart_type,  # Use 'z' or 'y' to control impedance vs admittance
        draw_labels=True,
        draw_vswr=True,
        show_legend=True
    )

KEY POINTS:
1. Use chart_type='z' for impedance Smith chart (default)
2. Use chart_type='y' for admittance Smith chart
3. No draw_admittance parameter exists in scikit-rf
4. No legend_location parameter in Network.plot_s_smith()
5. GUI chart_type_combo has values "z (Impedance)" and "y (Admittance)"
6. Extract just 'z' or 'y' from the combo box before passing to plot_s_smith()
"""
