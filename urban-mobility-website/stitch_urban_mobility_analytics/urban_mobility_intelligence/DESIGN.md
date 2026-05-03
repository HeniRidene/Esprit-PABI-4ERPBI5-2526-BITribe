---
name: Urban Mobility Intelligence
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#464651'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#777682'
  outline-variant: '#c7c5d3'
  surface-tint: '#5256a7'
  primary: '#000018'
  on-primary: '#ffffff'
  primary-container: '#0a0b63'
  on-primary-container: '#797dd1'
  inverse-primary: '#bfc1ff'
  secondary: '#006b5a'
  on-secondary: '#ffffff'
  secondary-container: '#79f8da'
  on-secondary-container: '#00725f'
  tertiary: '#030200'
  on-tertiary: '#ffffff'
  tertiary-container: '#281a00'
  on-tertiary-container: '#ad7c00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e1e0ff'
  primary-fixed-dim: '#bfc1ff'
  on-primary-fixed: '#080962'
  on-primary-fixed-variant: '#3a3e8d'
  secondary-fixed: '#79f8da'
  secondary-fixed-dim: '#59dbbf'
  on-secondary-fixed: '#00201a'
  on-secondary-fixed-variant: '#005143'
  tertiary-fixed: '#ffdea8'
  tertiary-fixed-dim: '#ffba20'
  on-tertiary-fixed: '#271900'
  on-tertiary-fixed-variant: '#5e4200'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-kpi:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: '1.4'
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.6'
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: 0.05em
  metric-trend:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 32px
  xl: 48px
  gutter: 20px
  margin-page: 32px
---

## Brand & Style
The design system is engineered for high-stakes decision-making in urban infrastructure and public transit. It adopts a **Corporate Modern** aesthetic, emphasizing clarity, structural integrity, and executive-level data density. The brand personality is authoritative yet progressive—balancing the institutional reliability of deep navy with the forward-thinking energy of vibrant teals and greens. 

The visual language focuses on "airiness" to prevent data fatigue, using generous white space and soft depth to organize complex information hierarchies. It targets urban planners, government officials, and logistics analysts, evoking a sense of calm control over complex, real-time city metrics.

## Colors
The palette is rooted in a functional hierarchy. 
- **Primary (Deep Navy)**: Used for high-level navigation, headers, and primary actions to establish an institutional "anchor."
- **Success/Green (Vibrant Teal)**: Represents positive growth, "greening" initiatives, and targets met.
- **Warning (Amber)**: Signals emerging issues or areas requiring attention.
- **Critical (Bright Red)**: Reserved exclusively for system failures or safety violations.
- **Grays**: A light-gray foundation (`#F8FAFC`) allows white cards to pop, creating a clear distinction between the workspace and individual data modules.

## Typography
This design system utilizes **Inter** for its exceptional legibility at small sizes and its clean, neutral character. 
- **KPI Hierarchy**: Large metrics use `display-kpi` to ensure immediate impact.
- **Labels**: Supporting labels utilize `label-caps` to provide context without competing visually with the data itself.
- **Color usage**: Headers should predominantly use the primary navy to maintain brand consistency, while body text uses a softened charcoal for long-form readability.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a 12-column structure for main dashboard views. 
- **Rhythm**: A 4px baseline grid ensures vertical consistency across tables and forms.
- **Gaps**: Components are separated by a 24px (md) gutter to maintain the "airy" feel.
- **Sidebars**: Filters and secondary navigation are housed in a fixed-width left rail (approx. 280px) while the primary data canvas expands to fill the viewport.

## Elevation & Depth
Depth is used functionally to separate the "dashboard canvas" from "interactive cards."
- **Base Layer**: The background is flat and light gray.
- **Card Layer**: All data modules sit on white containers with a **Soft Ambient Shadow** (0px 4px 20px rgba(0, 0, 0, 0.05)).
- **Interactive Layer**: Elements like dropdowns or modals use a more pronounced shadow to indicate they are "floating" above the dashboard.
- **Borders**: Card-level borders are avoided in favor of shadow-defined edges to keep the UI modern and clean.

## Shapes
The design system employs a consistent **Rounded (0.5rem)** corner radius for standard cards, buttons, and input fields. 
- **Large containers**: Sections and primary dashboard cards use a larger 1rem radius to soften the technical nature of the data.
- **Indicators**: Status dots and gauges utilize perfect circles to contrast against the rectangular grid.
- **Charts**: Bars and trend lines should feature slightly rounded end-caps to maintain the approachable mood.

## Components
- **KPI Cards**: Feature a large numerical value, a label, and a color-coded sparkline (Teal for positive, Red for negative trends) aligned to the right.
- **Circular Gauges**: Used for indices like AQI; they feature a multi-colored track (green-to-red) with a centralized needle or value.
- **Operational Tables**: High-density rows with subtle 1px dividers. Use status chips (rounded-pill) for categorical data.
- **Filters**: Located in the left sidebar, utilizing checkboxes and custom-styled dropdowns with "Apply" buttons fixed at the bottom.
- **Sparklines**: Ultra-simplified line charts without axes, used as visual "glanceable" indicators of directionality.
- **Budget Blocks**: Treemap-style components using solid fills of the primary and secondary colors to show proportional allocation.