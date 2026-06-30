"""Extracted from MainWindow: CaseSlidersMixin."""

from __future__ import annotations

from typing import List, Optional, Tuple

from holoshed.ui.qt import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget

class CaseSlidersMixin:

    def _update_case_sliders(self) -> None:
        """Create/update per-case time sliders when multiple datasets are loaded."""
        # Update both 1D and 2D sliders
        self._update_case_sliders_1d()
        self._update_case_sliders_2d()

    def _update_case_sliders_1d(self) -> None:
        """Create/update per-case time sliders for 1D mode."""
        # Determine which labels should have per-case sliders (all except the first)
        case_labels = list(self.cases.keys())
        labels_needing_sliders = set(case_labels[1:]) if len(case_labels) > 1 else set()

        # Remove sliders for cases that no longer exist or became the first case
        for label in list(self._case_slider_widgets.keys()):
            if label not in labels_needing_sliders:
                widget = self._case_slider_widgets.pop(label, None)
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                self._case_sliders.pop(label, None)
                self._case_spinboxes.pop(label, None)
                self._case_ms_spinboxes.pop(label, None)
                self._case_time_indices.pop(label, None)

        # Only show per-case sliders when multiple 1D datasets are loaded
        show_sliders = len(self.cases) > 1 and not self._mode_is_2d
        self._case_sliders_container.setVisible(show_sliders)

        # Update main slider label (show first dataset name when multiple loaded)
        if show_sliders and case_labels:
            first_label = case_labels[0]
            short_label = first_label if len(first_label) <= 20 else first_label[:17] + "..."
            self._main_slider_label.setText(short_label)
            self._main_slider_label.setToolTip(first_label)
            self._main_slider_label.setVisible(True)
        else:
            self._main_slider_label.setVisible(False)

        if not show_sliders:
            return

        # Add/update sliders for each case (skip first - it uses the main slider)
        case_items = list(self.cases.items())
        for label, c in case_items[1:]:
            if label in self._case_slider_widgets:
                # Update existing slider range
                slider = self._case_sliders[label]
                slider.blockSignals(True)
                slider.setMaximum(max(0, c.n_time - 1))
                # Clamp current value to valid range
                current = self._case_time_indices.get(label, c.n_time - 1)
                current = min(current, c.n_time - 1)
                slider.setValue(current)
                self._case_time_indices[label] = current
                slider.blockSignals(False)
                # Also update the spinbox
                spinbox = self._case_spinboxes.get(label)
                if spinbox is not None:
                    spinbox.blockSignals(True)
                    spinbox.setRange(0, max(0, c.n_time - 1))
                    spinbox.setValue(current)
                    spinbox.blockSignals(False)
                # Update the ms spinbox value
                ms_spin = self._case_ms_spinboxes.get(label)
                if ms_spin is not None:
                    t_ms = self._get_time_ms_for_case(c, current)
                    ms_spin.blockSignals(True)
                    ms_spin.setValue(t_ms)
                    ms_spin.blockSignals(False)
            else:
                # Create new slider row
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_layout.setSpacing(4)

                # Short label (truncate if needed)
                short_label = label if len(label) <= 20 else label[:17] + "..."
                lbl = QLabel(short_label)
                lbl.setToolTip(label)
                lbl.setMinimumWidth(60)
                lbl.setMaximumWidth(120)
                row_layout.addWidget(lbl)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(max(0, c.n_time - 1))
                slider.setSingleStep(1)
                slider.setPageStep(1)
                # Initialize to final time step (like main slider)
                initial_val = c.n_time - 1
                slider.setValue(initial_val)
                self._case_time_indices[label] = initial_val
                row_layout.addWidget(slider, 1)

                # Index spinbox (matching main slider style)
                row_layout.addWidget(QLabel("idx"))
                spin = QSpinBox()
                spin.setRange(0, max(0, c.n_time - 1))
                spin.setValue(initial_val)
                spin.setMinimumWidth(50)
                spin.setMaximumWidth(60)
                try:
                    spin.setKeyboardTracking(False)
                except Exception:
                    pass
                row_layout.addWidget(spin)

                # Time (ms) spinbox
                row_layout.addWidget(QLabel("t [ms]"))
                ms_spin = QDoubleSpinBox()
                ms_spin.setDecimals(4)
                ms_spin.setSingleStep(0.1)
                ms_spin.setRange(0.0, 1.0e12)
                t_ms = self._get_time_ms_for_case(c, initial_val)
                ms_spin.setValue(t_ms)
                ms_spin.setMinimumWidth(70)
                ms_spin.setMaximumWidth(90)
                try:
                    ms_spin.setKeyboardTracking(False)
                except Exception:
                    pass
                row_layout.addWidget(ms_spin)

                # Connect signals
                def make_slider_handler(case_label, spinbox, ms_spinbox, case_obj):
                    def handler(v):
                        self._case_time_indices[case_label] = int(v)
                        spinbox.blockSignals(True)
                        spinbox.setValue(int(v))
                        spinbox.blockSignals(False)
                        # Update ms spinbox
                        t_ms = self._get_time_ms_for_case(case_obj, int(v))
                        ms_spinbox.blockSignals(True)
                        ms_spinbox.setValue(t_ms)
                        ms_spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                def make_spin_handler(case_label, sldr, ms_spinbox, case_obj):
                    def handler(v):
                        self._case_time_indices[case_label] = int(v)
                        sldr.blockSignals(True)
                        sldr.setValue(int(v))
                        sldr.blockSignals(False)
                        # Update ms spinbox
                        t_ms = self._get_time_ms_for_case(case_obj, int(v))
                        ms_spinbox.blockSignals(True)
                        ms_spinbox.setValue(t_ms)
                        ms_spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                def make_ms_spin_handler(case_label, sldr, spinbox, case_obj):
                    def handler(v):
                        # Find nearest time index for this ms value
                        ti = self._find_time_index_for_ms(case_obj, v)
                        self._case_time_indices[case_label] = ti
                        sldr.blockSignals(True)
                        sldr.setValue(ti)
                        sldr.blockSignals(False)
                        spinbox.blockSignals(True)
                        spinbox.setValue(ti)
                        spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                slider.valueChanged.connect(make_slider_handler(label, spin, ms_spin, c))
                spin.valueChanged.connect(make_spin_handler(label, slider, ms_spin, c))
                ms_spin.valueChanged.connect(make_ms_spin_handler(label, slider, spin, c))

                self._case_sliders[label] = slider
                self._case_spinboxes[label] = spin
                self._case_ms_spinboxes[label] = ms_spin
                self._case_slider_widgets[label] = row_widget
                self._case_sliders_layout.addWidget(row_widget)

    def _update_case_sliders_2d(self) -> None:
        """Create/update per-case time sliders for 2D mode."""
        # Determine which labels should have per-case sliders (all except the first)
        case_labels = list(self.cases.keys())
        labels_needing_sliders = set(case_labels[1:]) if len(case_labels) > 1 else set()

        # Remove sliders for cases that no longer exist or became the first case
        for label in list(self._case_slider_widgets_2d.keys()):
            if label not in labels_needing_sliders:
                widget = self._case_slider_widgets_2d.pop(label, None)
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                self._case_sliders_2d.pop(label, None)
                self._case_spinboxes_2d.pop(label, None)
                self._case_ms_spinboxes_2d.pop(label, None)
                self._case_time_indices_2d.pop(label, None)

        # Only show per-case sliders when multiple 2D datasets are loaded
        show_sliders = len(self.cases) > 1 and self._mode_is_2d
        self._case_sliders_container_2d.setVisible(show_sliders)

        # Update main 2D slider label (show first dataset name when multiple loaded)
        if show_sliders and case_labels:
            first_label = case_labels[0]
            short_label = first_label if len(first_label) <= 20 else first_label[:17] + "..."
            self._main_slider_label_2d.setText(short_label)
            self._main_slider_label_2d.setToolTip(first_label)
            self._main_slider_label_2d.setVisible(True)
        else:
            self._main_slider_label_2d.setVisible(False)

        if not show_sliders:
            return

        # Add/update sliders for each case (skip first - it uses the main slider)
        case_items = list(self.cases.items())
        for label, c in case_items[1:]:
            if label in self._case_slider_widgets_2d:
                # Update existing slider range
                slider = self._case_sliders_2d[label]
                slider.blockSignals(True)
                slider.setMaximum(max(0, c.n_time - 1))
                # Clamp current value to valid range
                current = self._case_time_indices_2d.get(label, c.n_time - 1)
                current = min(current, c.n_time - 1)
                slider.setValue(current)
                self._case_time_indices_2d[label] = current
                slider.blockSignals(False)
                # Also update the spinbox
                spinbox = self._case_spinboxes_2d.get(label)
                if spinbox is not None:
                    spinbox.blockSignals(True)
                    spinbox.setRange(0, max(0, c.n_time - 1))
                    spinbox.setValue(current)
                    spinbox.blockSignals(False)
                # Update the ms spinbox value
                ms_spin = self._case_ms_spinboxes_2d.get(label)
                if ms_spin is not None:
                    t_ms = self._get_time_ms_for_case(c, current)
                    ms_spin.blockSignals(True)
                    ms_spin.setValue(t_ms)
                    ms_spin.blockSignals(False)
            else:
                # Create new slider row
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_layout.setSpacing(4)

                # Short label (truncate if needed)
                short_label = label if len(label) <= 20 else label[:17] + "..."
                lbl = QLabel(short_label)
                lbl.setToolTip(label)
                lbl.setMinimumWidth(60)
                lbl.setMaximumWidth(120)
                row_layout.addWidget(lbl)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(max(0, c.n_time - 1))
                slider.setSingleStep(1)
                slider.setPageStep(1)
                # Initialize to final time step (like main slider)
                initial_val = c.n_time - 1
                slider.setValue(initial_val)
                self._case_time_indices_2d[label] = initial_val
                row_layout.addWidget(slider, 1)

                # Index spinbox (matching main slider style)
                row_layout.addWidget(QLabel("idx"))
                spin = QSpinBox()
                spin.setRange(0, max(0, c.n_time - 1))
                spin.setValue(initial_val)
                spin.setMinimumWidth(50)
                spin.setMaximumWidth(60)
                try:
                    spin.setKeyboardTracking(False)
                except Exception:
                    pass
                row_layout.addWidget(spin)

                # Time (ms) spinbox
                row_layout.addWidget(QLabel("t [ms]"))
                ms_spin = QDoubleSpinBox()
                ms_spin.setDecimals(4)
                ms_spin.setSingleStep(0.1)
                ms_spin.setRange(0.0, 1.0e12)
                t_ms = self._get_time_ms_for_case(c, initial_val)
                ms_spin.setValue(t_ms)
                ms_spin.setMinimumWidth(70)
                ms_spin.setMaximumWidth(90)
                try:
                    ms_spin.setKeyboardTracking(False)
                except Exception:
                    pass
                row_layout.addWidget(ms_spin)

                # Connect signals (use 2D-specific dicts)
                def make_slider_handler_2d(case_label, spinbox, ms_spinbox, case_obj):
                    def handler(v):
                        self._case_time_indices_2d[case_label] = int(v)
                        spinbox.blockSignals(True)
                        spinbox.setValue(int(v))
                        spinbox.blockSignals(False)
                        # Update ms spinbox
                        t_ms = self._get_time_ms_for_case(case_obj, int(v))
                        ms_spinbox.blockSignals(True)
                        ms_spinbox.setValue(t_ms)
                        ms_spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                def make_spin_handler_2d(case_label, sldr, ms_spinbox, case_obj):
                    def handler(v):
                        self._case_time_indices_2d[case_label] = int(v)
                        sldr.blockSignals(True)
                        sldr.setValue(int(v))
                        sldr.blockSignals(False)
                        # Update ms spinbox
                        t_ms = self._get_time_ms_for_case(case_obj, int(v))
                        ms_spinbox.blockSignals(True)
                        ms_spinbox.setValue(t_ms)
                        ms_spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                def make_ms_spin_handler_2d(case_label, sldr, spinbox, case_obj):
                    def handler(v):
                        # Find nearest time index for this ms value
                        ti = self._find_time_index_for_ms(case_obj, v)
                        self._case_time_indices_2d[case_label] = ti
                        sldr.blockSignals(True)
                        sldr.setValue(ti)
                        sldr.blockSignals(False)
                        spinbox.blockSignals(True)
                        spinbox.setValue(ti)
                        spinbox.blockSignals(False)
                        self.request_redraw()
                    return handler

                slider.valueChanged.connect(make_slider_handler_2d(label, spin, ms_spin, c))
                spin.valueChanged.connect(make_spin_handler_2d(label, slider, ms_spin, c))
                ms_spin.valueChanged.connect(make_ms_spin_handler_2d(label, slider, spin, c))

                self._case_sliders_2d[label] = slider
                self._case_spinboxes_2d[label] = spin
                self._case_ms_spinboxes_2d[label] = ms_spin
                self._case_slider_widgets_2d[label] = row_widget
                self._case_sliders_layout_2d.addWidget(row_widget)

