from typing import List, Dict
from dataclasses import dataclass

# Color palettes as constants
COLOR_PALETTES: Dict[str, List[str]] = {
    'origin': [
        '#515151',  # grey
        '#F14040',  # red
        '#1A6FDF',  # blue
        '#37AD6B',  # green
        '#B177DE',  # purple
        '#FEC211',  # yellow
        '#999999',  # grey
        '#FF4081',  # hot pink
        '#FF6666',  # baby pink
        '#FB6501',  # orange
        '#6699CC',  # light blue
        '#97D7F2',  # light blue
        '#EAA5C2',  # light pink
        '#B3D49D',  # light green
        '#BdBeDC',  # light purple
        '#F6EDAA',  # light yellow
        '#CC9900',  # yellow
        '#00CBCC',  # cyan
        '#7D4E4E',  # brown
        '#8E8E00',  # olive
        '#6FB802',  # light green
        '#07AEE3',  # light blue 
    ]
}

@dataclass
class PlotConfig:
    """Configuration for plotting parameters.
    
    Attributes:
        colors (List[str]): List of colors for plotting
        font_size (int): Base font size for plots
        label_size (int): Font size for labels
        line_width (float): Width of plot lines
        
    Example:
        >>> config = PlotConfig()
        >>> plt.plot(x, y, color=config.colors[0], linewidth=config.line_width)
    """
    colors: List[str] = COLOR_PALETTES['origin']
    font_size: int = 8
    label_size: int = 10
    line_width: float = 0.5
    
    @classmethod
    def with_palette(cls, palette_name: str) -> 'PlotConfig':
        """Create PlotConfig with a specific color palette.
        
        Args:
            palette_name (str): Name of the color palette to use
            
        Returns:
            PlotConfig: New configuration instance with specified palette
            
        Raises:
            KeyError: If palette_name is not found
        """
        if palette_name not in COLOR_PALETTES:
            raise KeyError(f"Color palette '{palette_name}' not found")
        return cls(colors=COLOR_PALETTES[palette_name])



