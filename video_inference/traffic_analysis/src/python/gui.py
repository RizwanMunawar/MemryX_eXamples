"""
Traffic Analysis GUI - Modern PyQt5 Interface
Features: Zone drawing with mouse, color wheel picker, shutdown handling
"""
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QColorDialog, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from dataclasses import dataclass


@dataclass
class ZoneData:
    """Data class for zone information"""
    name: str
    rect: QRect
    color: QColor


class ColorWheelButton(QPushButton):
    """Custom button that shows current color and opens color dialog"""
    color_changed = pyqtSignal(QColor)
    
    def __init__(self, initial_color=QColor(255, 180, 0)):
        super().__init__()
        self._color = initial_color
        self.setFixedSize(45, 45)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._open_color_dialog)
        self._update_style()
        
    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name()};
                border: 2px solid #2a2a3e;
                border-radius: 22px;
            }}
            QPushButton:hover {{
                border: 2px solid #ffffff;
            }}
        """)
        
    def _open_color_dialog(self):
        color = QColorDialog.getColor(
            self._color, 
            self, 
            "Select Zone Color",
            QColorDialog.ShowAlphaChannel
        )
        if color.isValid():
            self._color = color
            self._update_style()
            self.color_changed.emit(color)
    
    def color(self):
        return self._color
    
    def set_color(self, color):
        self._color = color
        self._update_style()


class VideoCanvas(QLabel):
    """Canvas for video display and zone drawing"""
    zone_created = pyqtSignal(QRect)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #0a0a12;
                border: 2px solid #2a2a3e;
                border-radius: 8px;
            }
        """)
        
        self._drawing = False
        self._start_point = QPoint()
        self._end_point = QPoint()
        self._current_color = QColor(255, 180, 0)
        self._zones = []
        self._base_pixmap = None
        self._draw_mode = False
        self._image_rect = QRect()
        
    def set_draw_mode(self, enabled):
        self._draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        
    def set_current_color(self, color):
        self._current_color = color
        
    def set_zones(self, zones):
        self._zones = zones
        self._update_display()
        
    def update_frame(self, frame):
        """Update display with new video frame (numpy array BGR)"""
        if frame is None:
            return
            
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self._base_pixmap = QPixmap.fromImage(q_img)
        self._update_display()
        
    def _update_display(self):
        if self._base_pixmap is None:
            return
            
        scaled = self._base_pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        self._image_rect = QRect(x_offset, y_offset, scaled.width(), scaled.height())
        
        display = QPixmap(self.size())
        display.fill(QColor(10, 10, 18))
        
        painter = QPainter(display)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(self._image_rect.topLeft(), scaled)
        
        scale_x = scaled.width() / self._base_pixmap.width()
        scale_y = scaled.height() / self._base_pixmap.height()
        
        for zone in self._zones:
            scaled_rect = QRect(
                int(zone.rect.x() * scale_x) + x_offset,
                int(zone.rect.y() * scale_y) + y_offset,
                int(zone.rect.width() * scale_x),
                int(zone.rect.height() * scale_y)
            )
            
            fill_color = QColor(zone.color)
            fill_color.setAlpha(80)
            painter.fillRect(scaled_rect, fill_color)
            
            pen = QPen(zone.color, 2)
            painter.setPen(pen)
            painter.drawRect(scaled_rect)
            
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.setPen(QPen(Qt.white))
            text_rect = scaled_rect.adjusted(5, 5, -5, -5)
            painter.drawText(text_rect, Qt.AlignTop | Qt.AlignLeft, zone.name)
        
        if self._drawing:
            rect = QRect(self._start_point, self._end_point).normalized()
            fill = QColor(self._current_color)
            fill.setAlpha(50)
            painter.fillRect(rect, fill)
            painter.setPen(QPen(self._current_color, 2, Qt.DashLine))
            painter.drawRect(rect)
            
        painter.end()
        self.setPixmap(display)
        
    def _to_image_coords(self, point):
        """Convert widget coords to image coords"""
        if self._base_pixmap is None or not self._image_rect.contains(point):
            return None
            
        scale_x = self._base_pixmap.width() / self._image_rect.width()
        scale_y = self._base_pixmap.height() / self._image_rect.height()
        
        return QPoint(
            int((point.x() - self._image_rect.x()) * scale_x),
            int((point.y() - self._image_rect.y()) * scale_y)
        )
        
    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton:
            if self._image_rect.contains(event.pos()):
                self._drawing = True
                self._start_point = event.pos()
                self._end_point = event.pos()
                
    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end_point = event.pos()
            self._update_display()
            
    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            self._end_point = event.pos()
            
            # Convert to image coordinates
            start_img = self._to_image_coords(self._start_point)
            end_img = self._to_image_coords(self._end_point)
            
            if start_img and end_img:
                rect = QRect(start_img, end_img).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    self.zone_created.emit(rect)
                    
            self._update_display()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ZoneListItem(QWidget):
    """Custom widget for zone list items"""
    delete_clicked = pyqtSignal(str)
    
    def __init__(self, zone_data):
        super().__init__()
        self.zone_data = zone_data
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        color_label = QLabel()
        color_label.setFixedSize(16, 16)
        color_label.setStyleSheet(f"""
            background-color: {self.zone_data.color.name()};
            border-radius: 8px;
            border: 1px solid #444;
        """)
        
        name_label = QLabel(self.zone_data.name)
        name_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(22, 22)
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff6b6b;
            }
        """)
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.zone_data.name))
        
        layout.addWidget(color_label)
        layout.addWidget(name_label, 1)
        layout.addWidget(delete_btn)


class StatsPanel(QFrame):
    """Panel showing zone statistics"""
    def __init__(self):
        super().__init__()
        # Preferred policy works well with scrollarea's widgetResizable
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
        
    def _setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 8px;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("📊 Zone Statistics")
        title.setStyleSheet("""
            color: #ffffff;
            font-size: 13px;
            font-weight: bold;
            padding-bottom: 5px;
        """)
        self.layout.addWidget(title)
        
        self.stats_container = QVBoxLayout()
        self.stats_container.setSpacing(6)
        self.layout.addLayout(self.stats_container)
        self.layout.addStretch()  # Push content to top when empty
        
    def update_stats(self, zones_data):
        while self.stats_container.count():
            child = self.stats_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        for zone in zones_data:
            if not zone.get('incoming_counts'):
                continue
                
            zone_frame = QFrame()
            zone_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            zone_frame.setStyleSheet("""
                QFrame {
                    background-color: #252540;
                    border-radius: 6px;
                }
            """)
            zone_layout = QVBoxLayout(zone_frame)
            zone_layout.setSpacing(6)
            zone_layout.setContentsMargins(10, 10, 10, 10)
            
            name_label = QLabel(zone['name'])
            # Use the zone's own color for its name
            zone_color = zone.get('color', (76, 201, 240))  # Default to sky blue if not set
            zone_color_hex = f"#{zone_color[2]:02x}{zone_color[1]:02x}{zone_color[0]:02x}"
            name_label.setStyleSheet(f"color: {zone_color_hex}; font-weight: bold; font-size: 14px;")
            name_label.setMinimumHeight(20)
            zone_layout.addWidget(name_label)
            
            # Horizontal row of entries:  ← Zone_X: N   ← Zone_Y: M   ...
            row_layout = QHBoxLayout()
            row_layout.setSpacing(20)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            for src, count in zone['incoming_counts'].items():
                src_color = zone['source_colors'].get(src, (100, 100, 100))
                color_hex = f"#{src_color[2]:02x}{src_color[1]:02x}{src_color[0]:02x}"
                
                label = QLabel(f"← {src}: {count}")
                label.setStyleSheet(f"color: {color_hex}; font-size: 13px;")
                label.setMinimumHeight(18)
                row_layout.addWidget(label)
            
            row_layout.addStretch()
            zone_layout.addLayout(row_layout)
            self.stats_container.addWidget(zone_frame)


class TrafficAnalysisGUI(QMainWindow):
    """Main GUI Window"""
    shutdown_requested = pyqtSignal()
    zones_changed = pyqtSignal(object)  # emits List[ZoneData]
    
    def __init__(self):
        super().__init__()
        self._zones = []
        self._zone_counter = 1
        self._analysis_running = False
        self._setup_ui()
        
    def _setup_ui(self):
        self.setWindowTitle("🚦 Traffic Analysis System")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet(self._get_stylesheet())
        
        central = QWidget()
        self.setCentralWidget(central)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(3)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #2a2a3e;
            }
            QSplitter::handle:hover {
                background-color: #3a3a4e;
            }
        """)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)
        main_layout.addWidget(main_splitter)
        
        video_panel = QWidget()
        video_panel.setMinimumWidth(600)
        video_layout = QVBoxLayout(video_panel)
        video_layout.setSpacing(10)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = VideoCanvas()
        self.canvas.zone_created.connect(self._on_zone_created)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(640, 480)
        video_layout.addWidget(self.canvas, 1)
        status_bar = QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("""
            color: #4cc9f0;
            font-size: 14px;
            font-weight: bold;
            padding: 5px 15px;
        """)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("""
            color: #ffffff;
            font-size: 13px;
            padding: 5px 15px;
        """)
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)
        
        video_layout.addWidget(status_bar)
        main_splitter.addWidget(video_panel)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel(" Traffic Analysis")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 20px;
            font-weight: bold;
            padding: 8px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)
        
        # Zone Drawing Controls
        zone_group = QGroupBox(" Zone Drawing")
        zone_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        zone_group.setStyleSheet("""
            QGroupBox {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #2a2a3e;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)
        zone_layout = QVBoxLayout(zone_group)
        zone_layout.setSpacing(10)
        zone_layout.setContentsMargins(12, 12, 12, 12)
        
        color_row = QHBoxLayout()
        color_label = QLabel("Zone Color:")
        color_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: normal;")
        color_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.color_picker = ColorWheelButton(QColor(255, 180, 0))
        self.color_picker.color_changed.connect(self._on_color_changed)
        color_row.addWidget(color_label)
        color_row.addStretch()
        color_row.addWidget(self.color_picker)
        zone_layout.addLayout(color_row)
        
        self.draw_btn = QPushButton("🖊️  Draw Zone")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setCursor(Qt.PointingHandCursor)
        self.draw_btn.setMinimumHeight(40)
        self.draw_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.draw_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4361ee, stop:1 #3a0ca3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5371ff, stop:1 #4a1cb3);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f72585, stop:1 #b5179e);
            }
        """)
        self.draw_btn.toggled.connect(self._on_draw_mode_toggled)
        zone_layout.addWidget(self.draw_btn)
        
        zone_list_label = QLabel("Active Zones:")
        zone_list_label.setStyleSheet("color: #888; font-size: 11px; font-weight: normal; margin-top: 5px;")
        zone_layout.addWidget(zone_list_label)
        
        self.zone_list = QListWidget()
        self.zone_list.setMinimumHeight(100)
        self.zone_list.setMaximumHeight(180)
        self.zone_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.zone_list.setStyleSheet("""
            QListWidget {
                background-color: #12121e;
                border: 1px solid #2a2a3e;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                background-color: transparent;
                border-radius: 4px;
                padding: 2px;
            }
            QListWidget::item:selected {
                background-color: #2a2a3e;
            }
        """)
        zone_layout.addWidget(self.zone_list)
        
        self.clear_zones_btn = QPushButton("🗑️  Clear All Zones")
        self.clear_zones_btn.setCursor(Qt.PointingHandCursor)
        self.clear_zones_btn.setMinimumHeight(35)
        self.clear_zones_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.clear_zones_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3e;
                color: #aaa;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a4e;
                color: #fff;
            }
        """)
        self.clear_zones_btn.clicked.connect(self._clear_zones)
        zone_layout.addWidget(self.clear_zones_btn)
        
        right_layout.addWidget(zone_group)
        
        self.stats_panel = StatsPanel()
        stats_scroll = QScrollArea()
        stats_scroll.setWidget(self.stats_panel)
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setAlignment(Qt.AlignTop)
        stats_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a2e;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background-color: #252540;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a4a6a;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5a5a7a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        stats_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_layout.addWidget(stats_scroll, 1)
        
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.start_btn = QPushButton("▶️  Start Analysis")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b894, stop:1 #00a085);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d9a5, stop:1 #00b894);
            }
            QPushButton:disabled {
                background: #2a2a3e;
                color: #666;
            }
        """)
        buttons_layout.addWidget(self.start_btn)
        
        self.shutdown_btn = QPushButton("⏹️  Shutdown")
        self.shutdown_btn.setCursor(Qt.PointingHandCursor)
        self.shutdown_btn.setMinimumHeight(45)
        self.shutdown_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.shutdown_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b5b, stop:1 #e74c3c);
            }
        """)
        self.shutdown_btn.clicked.connect(self._on_shutdown)
        buttons_layout.addWidget(self.shutdown_btn)
        
        right_layout.addLayout(buttons_layout)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([700, 300])
        
    def _get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #0f0f1a;
            }
            QWidget {
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #1a1a2e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3a4e;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4a4a5e;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        
    def _on_color_changed(self, color):
        self.canvas.set_current_color(color)
        
    def _on_draw_mode_toggled(self, checked):
        self.canvas.set_draw_mode(checked)
        if checked:
            self.draw_btn.setText("✏️  Drawing... (click to stop)")
            self.status_label.setText("● Draw Mode: Click and drag to create zones")
            self.status_label.setStyleSheet("""
                color: #f72585;
                font-size: 14px;
                font-weight: bold;
                padding: 5px 15px;
            """)
        else:
            self.draw_btn.setText("🖊️  Draw Zone")
            self.status_label.setText("● Ready")
            self.status_label.setStyleSheet("""
                color: #4cc9f0;
                font-size: 14px;
                font-weight: bold;
                padding: 5px 15px;
            """)
            
    def _on_zone_created(self, rect):
        name = f"Zone_{self._zone_counter}"
        self._zone_counter += 1
        
        zone = ZoneData(name, rect, QColor(self.color_picker.color()))
        self._zones.append(zone)
        self._update_zone_list()
        self.canvas.set_zones(self._zones)
        self.zones_changed.emit(self._zones)
        
        colors = [
            QColor(255, 180, 0),
            QColor(90, 90, 255),
            QColor(100, 220, 100),
            QColor(255, 120, 200),
            QColor(100, 200, 255),
            QColor(180, 100, 255),
        ]
        next_idx = len(self._zones) % len(colors)
        self.color_picker.set_color(colors[next_idx])
        self.canvas.set_current_color(colors[next_idx])
        
    def _update_zone_list(self):
        self.zone_list.clear()
        for zone in self._zones:
            item = QListWidgetItem()
            widget = ZoneListItem(zone)
            widget.delete_clicked.connect(self._delete_zone)
            item.setSizeHint(widget.sizeHint())
            self.zone_list.addItem(item)
            self.zone_list.setItemWidget(item, widget)
            
    def _delete_zone(self, name):
        self._zones = [z for z in self._zones if z.name != name]
        self._update_zone_list()
        self.canvas.set_zones(self._zones)
        self.zones_changed.emit(self._zones)
        
    def _clear_zones(self):
        self._zones = []
        self._zone_counter = 1
        self._update_zone_list()
        self.canvas.set_zones(self._zones)
        self.zones_changed.emit(self._zones)
        
    def _on_shutdown(self):
        self.shutdown_requested.emit()
        self.close()
        
    def start_analysis(self):
        """Toggle analysis state"""
        self._analysis_running = not self._analysis_running
        
        if self._analysis_running:
            self.start_btn.setText("⏸️  Pause Analysis")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f39c12, stop:1 #e67e22);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 15px 25px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f7b731, stop:1 #f39c12);
                }
            """)
            self.status_label.setText("● Analyzing...")
            self.status_label.setStyleSheet("""
                color: #00b894;
                font-size: 14px;
                font-weight: bold;
                padding: 5px 15px;
            """)
        else:
            self.start_btn.setText("▶️  Resume Analysis")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00b894, stop:1 #00a085);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 15px 25px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00d9a5, stop:1 #00b894);
                }
            """)
            self.status_label.setText("● Paused")
            self.status_label.setStyleSheet("""
                color: #f39c12;
                font-size: 14px;
                font-weight: bold;
                padding: 5px 15px;
            """)
            
    def is_analysis_running(self):
        return self._analysis_running
        
    def get_zones(self):
        return self._zones
        
    def update_frame(self, frame):
        self.canvas.update_frame(frame)
        
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    def update_zone_stats(self, zones_data):
        self.stats_panel.update_stats(zones_data)
        
    def closeEvent(self, event):
        self.shutdown_requested.emit()
        event.accept()
