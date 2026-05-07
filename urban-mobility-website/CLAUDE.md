# Stitch Urban Mobility Analytics - Complete Documentation

## Overview
The `stitch_urban_mobility_analytics` folder contains a comprehensive suite of Business Intelligence dashboards for urban mobility analytics. These are interactive, responsive web applications built with Tailwind CSS and Material Design 3 components, designed for real-time decision-making across urban infrastructure and public transit management.

**Target Audience**: Urban planners, government officials, logistics analysts, infrastructure managers, and sustainability coordinators.

---

## Folder Structure

```
stitch_urban_mobility_analytics/
├── live_mobility_map/                      # Real-time mobility tracking
│   └── code.html
├── safety_maintenance_management/          # Operational safety & maintenance
│   └── code.html
├── strategic_dashboard/                    # Strategic territory overview
│   └── code.html
├── sustainability_analytics/               # Environmental impact analytics
│   └── code.html
└── urban_mobility_intelligence/            # Design system & specifications
    └── DESIGN.md
```

---

## Dashboard Modules

### 1. **Live Mobility Map** (`live_mobility_map/code.html`)
**Purpose**: Real-time visualization of urban mobility patterns and transit flow.

**Key Features**:
- Interactive map layers visualization
- Traffic flow monitoring
- Transit line tracking
- Environmental sensor overlays
- Live incident reporting
- Dynamic layer toggling (Traffic Flow, Transit Lines, Environmental Sensors, Incidents)

**Components**:
- Map visualization canvas
- Layer control panel
- Real-time alert system
- Navigation sidebar with Territory Overview, Time Analysis, Analytical Layers, Census Data
- Responsive grid-based layout

**Data Metrics Displayed**:
- City Congestion Index (e.g., 68/100)
- Performance trends
- Active alerts and incidents
- Map layer selection controls

---

### 2. **Safety & Maintenance Management** (`safety_maintenance_management/code.html`)
**Purpose**: Operational health monitoring, incident tracking, and infrastructure maintenance management.

**Key Features**:
- Active sensor network monitoring
- Real-time maintenance alerts
- Incident density mapping by zone
- Sensor reliability tracking
- Strategic action recommendations
- Weekly maintenance alert tracking

**Components**:
- KPI cards for sensor count and alerts
- Zone-based incident density table
- Sensor reliability gauges
- Strategic action dispatcher
- Download functionality for views
- Search operational data capability

**Key Metrics**:
- Total Active Sensors (e.g., 14,295)
- Weekly Maintenance Alerts (e.g., 342)
- Zone Analysis (Nantes Centre, Strasbourg Europe, Lyon Part-Dieu, Paris La Défense)
- Accident Density tracking
- Sensor Reliability percentages (65-94%)

**Data Structure**:
- Zone: Location identifier
- Accident Density: High/Medium/Low classification with km² units
- Sensor Reliability: Percentage-based gauge
- Strategic Action: Dispatch Crew, Review Logs, Schedule Check

---

### 3. **Strategic Dashboard** (`strategic_dashboard/code.html`)
**Purpose**: High-level overview of urban mobility performance and sustainability initiatives.

**Key Features**:
- Territory-wide performance metrics
- Carbon intensity monitoring
- Cost Analysis and Score KPIs
- Energy consumption tracking
- Budget allocation visualization (Treemap)
- Air Quality Index (AQI) gauge
- Safety and maintenance overview
- Multi-view navigation (Overview, LEZ Projects, Transit Network, SDG Analysis)

**Components**:
- 4-column KPI layout (Functionality Rate, Carbon Intensity, CSAT Score, Energy Costum)
- Performance trend line chart
- Budget allocation treemap
- AQI circular gauge
- Zone-based safety metrics table
- Multi-territory filtering (All territories, Paris Region, Aix-Marseille-Provence, Lyon Métropole, Lille Métropole, Bordeaux Métropole)
- Period selector (Last 12 months)

**Key Metrics**:
- Functionality Rate: 94.2% (+0.4%)
- Carbon Intensity: 12.4 g/km (-1.2%)
- CSAT Score: 4.2/5 (Stable)
- Energy Costum: 1.25 € (+2.2%)
- Budget Allocation: Sustainability, Energy, Transit initiatives
- Air Quality Index: 42 (GOOD)

---

### 4. **Sustainability & Emissions Analytics** (`sustainability_analytics/code.html`)
**Purpose**: Environmental impact tracking and sustainability performance monitoring across the transport network.

**Key Features**:
- CO2 reduction tracking
- Fleet electrification monitoring
- Energy efficiency metrics by transport mode
- Greening projects distribution
- Live sensor network status
- Carbon emissions trend visualization
- Real-time data updates (Live Data indicator)

**Components**:
- 4-column KPI header (Total CO2 Reduction, Fleet Electrification, Energy Efficiency Index, Active Green Projects)
- Carbon Emissions Trend chart (6-month view with 1M/12M/ALL toggles)
- Energy Efficiency by Mode breakdown
- Greening Projects Distribution (pie chart/treemap)
- Live Sensor Network table
- Filter panel with territory selection
- Multi-analytics layer support

**Key Metrics**:
- Total CO2 Reduction: 42.5 kT (↑ 12%)
- Fleet Electrification: 68% (↑ 5%)
- Energy Efficiency Index: 92.4 (— 0%)
- Active Green Projects: 14 (+ 2 New)
- Sensor Status: Active/Maint/Failed tracking
- Energy Efficiency by Mode: Electric Bus (85 kWh/km), Train Network (82 kWh/km), Métro Lines (49 kWh/km)

---

## Design System (`urban_mobility_intelligence/DESIGN.md`)

### Brand Philosophy
**Aesthetic**: Corporate Modern
**Tone**: Authoritative yet progressive
**Vision**: Calm control over complex, real-time city metrics

The design system balances institutional reliability with forward-thinking energy. It emphasizes:
- **Airiness**: Generous white space prevents data fatigue
- **Clarity**: High data density without overwhelming users
- **Structural Integrity**: Clear visual hierarchy for decision-making

### Color Palette

#### Primary Colors
- **Primary (Deep Navy)**: `#000018`
  - Used for navigation, headers, primary actions
  - Establishes institutional "anchor"
  - On Primary: `#ffffff`

- **Secondary (Vibrant Teal)**: `#006b5a`
  - Represents positive growth and "greening" initiatives
  - Success states and targets met
  - Secondary Container: `#79f8da`

- **Tertiary (Dark Brown)**: `#030200`
  - Supporting accent color
  - Tertiary Container: `#281a00`
  - Used for contrast in complex hierarchies

#### Functional Colors
- **Success/Green**: Teal variants
  - Primary Fixed: `#e1e0ff`
  - Primary Fixed Dim: `#bfc1ff`
  - Secondary Fixed: `#79f8da`
  - Secondary Fixed Dim: `#59dbbf`

- **Warning (Amber)**: Orange/Brown tones
  - Tertiary Fixed: `#ffdea8`
  - Tertiary Fixed Dim: `#ffba20`
  - Signals emerging issues

- **Critical (Bright Red)**: `#ba1a1a`
  - Error states and safety violations
  - Error Container: `#ffdad6`
  - On Error: `#ffffff`

#### Surface Colors
- **Background**: `#f8f9ff` (Light gray foundation)
- **Surface**: `#f8f9ff`
- **Surface Variants**:
  - Surface Container: `#e5eeff`
  - Surface Container Low: `#eff4ff`
  - Surface Container High: `#dce9ff`
  - Surface Container Highest: `#d3e4fe`
  - Surface Dim: `#cbdbf5`
  - Surface Bright: `#f8f9ff`

- **Inverse Surface**: `#213145` (Dark mode support)
- **Inverse On Surface**: `#eaf1ff`

#### Text & Outline
- **On Surface**: `#0b1c30` (Primary text)
- **On Surface Variant**: `#464651` (Secondary text)
- **Outline**: `#777682` (Borders/dividers)
- **Outline Variant**: `#c7c5d3` (Light borders)

### Typography

**Font Family**: Inter (exceptional legibility at small sizes, clean neutral character)

**Type Scale**:

| Style | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| **display-kpi** | 36px | 700 | 1.2 | Large KPI metrics |
| **headline-sm** | 18px | 600 | 1.4 | Section headers, titles |
| **body-md** | 14px | 400 | 1.6 | Main body text |
| **metric-trend** | 12px | 600 | 1.2 | Trend indicators, supporting metrics |
| **label-caps** | 11px | 700 | 1.2 | Labels, category tags, uppercase text |

**Letter Spacing**:
- display-kpi: -0.02em (tighter)
- label-caps: 0.05em (wider for uppercase emphasis)

### Layout & Spacing

**Grid System**: 12-column fluid grid for main dashboard views

**Spacing Values** (4px baseline grid):
| Token | Value | Usage |
|-------|-------|-------|
| base | 4px | Micro-spacing |
| xs | 8px | Tight spacing |
| sm | 16px | Standard padding |
| md | 24px | Component padding |
| lg | 32px | Section margins |
| xl | 48px | Large section gaps |
| gutter | 20px | Grid gaps |
| margin-page | 32px | Page margins |

**Layout Patterns**:
- **Sidebar Navigation**: Fixed-width left rail (~280px)
- **Primary Canvas**: Expands to fill remaining viewport
- **Top Navigation Bar**: Sticky header (72px height) with search, notifications, settings
- **KPI Grid**: 1-4 columns responsive (1 mobile, 2 tablet, 4 desktop)
- **Card Layout**: Consistent 24px (md) gutter between cards

### Elevation & Depth

**Shadow System**:
- **Soft Ambient Shadow**: `0px 4px 20px rgba(0, 0, 0, 0.05)`
  - Used for all data module cards
  - Creates subtle depth without heaviness

- **Base Layer**: Flat light gray background
- **Card Layer**: White containers on base
- **Interactive Layer**: More pronounced shadow for floating elements (dropdowns, modals)

**Borders**:
- Card-level borders: Avoided in favor of shadow-defined edges (modern, clean aesthetic)
- Dividers: 1px subtle lines between table rows
- Color: `outline-variant` (#c7c5d3) for light borders

### Corner Radius

**Rounded Values**:
| Token | Value | Usage |
|-------|-------|-------|
| DEFAULT | 0.25rem | Small UI elements |
| lg | 0.5rem | Cards, buttons, inputs |
| xl | 0.75rem | Large containers |
| full | 9999px | Pills, circles |

**Application**:
- Standard cards: 0.5rem
- Large containers/sections: 1rem
- Status dots/gauges: Perfect circles (9999px)
- Charts: Slightly rounded end-caps (approachable mood)

### Component Library

#### KPI Cards
- Large numerical value with custom typography (`display-kpi`)
- Descriptive label with `label-caps`
- Color-coded sparkline (right-aligned)
  - **Teal**: Positive trends
  - **Red**: Negative trends
- Metric-trend indicator (e.g., "+2.4%")
- Icon support (Material Symbols)
- Flex layout for alignment

```html
<!-- Example KPI Card Structure -->
<div class="bg-surface-container-lowest rounded-xl p-md shadow-[0_4px_20px_rgba(0,0,0,0.05)]">
  <div class="font-label-caps text-outline">TOTAL ACTIVE SENSORS</div>
  <div class="font-display-kpi text-on-surface">14,295</div>
  <div class="font-metric-trend text-secondary">+2.4%</div>
</div>
```

#### Circular Gauges (AQI, Indices)
- Multi-colored track (green-to-red gradient)
- Centralized needle or value display
- Used for indices like Air Quality Index (AQI)
- Color indicates severity
- Support for "GOOD", "MODERATE", "UNHEALTHY" states

#### Operational Tables
- High-density row layouts
- Subtle 1px dividers between rows
- Status chips (rounded pills) for categorical data
  - Example: "High (4.2/km²)", "Low (1.1km²)", "Medium (2.8/km²)"
- Alternating row backgrounds (optional) for readability
- Column alignment: Zone | Metric | Value | Action

#### Sparklines
- Ultra-simplified line charts without axes
- Visual "glanceable" indicators of directionality
- Rendered inline with KPI text
- Color-coded by trend
- Responsive to data changes

#### Status Chips
- Rounded pill-shaped badges
- Color-coded: Green (Active), Orange (Maintenance), Red (Failed)
- Compact label styling
- Used in sensor networks, incident status

#### Budget/Allocation Blocks (Treemap)
- Proportional area representation
- Solid fills using primary and secondary colors
- Hierarchical organization
- Labels with values (€, %, counts)
- Example: "Greening 2.8M€", "Energy 1.5M€", "Transit 0.4M€"

#### Filters & Sidebars
- Fixed-width left navigation (280px)
- Checkboxes and custom dropdowns
- Territory/Period selectors
- "Apply Filters" button fixed at sidebar bottom
- Responsive collapse for mobile
- Icons from Material Symbols

#### Top Navigation Bar
- Sticky header (72px)
- Logo/branding left
- Search bar center (with search icon)
- Notification, settings, profile icons right
- Blur backdrop effect (`backdrop-blur-md`)
- Responsive design

### Interaction Patterns

**Hover States**:
- Navigation items: Translate left (1px), light background
- Buttons: Opacity/color transition
- Cards: Subtle shadow enhancement
- Duration: 150ms cubic-bezier ease-in-out

**Focus States**:
- Input focus: 2px ring using `primary-container`
- Keyboard navigation support

**Dark Mode**:
- Full dark mode support via Tailwind class
- Inverse surface colors automatically applied
- Color adjustments for all text and backgrounds
- Smooth transitions

---

## Technologies & Stack

### Frontend Framework & Styling
- **Tailwind CSS**: Utility-first CSS framework
  - `container-queries` plugin for component-level responsiveness
  - `forms` plugin for form styling
  - Custom configuration with Material Design 3 colors

- **Custom Typography**: Inter font family (Google Fonts)
  - Web fonts: `family=Inter:wght@400;500;600;700;900`

### Icons
- **Material Symbols Outlined** (Google Fonts)
  - Weight: 100-700
  - Fill: 0-1
  - Grade: -25 to 200
  - Size: 24
  - Custom variation settings applied via CSS

### Responsive Design
- Mobile-first approach
- Breakpoints: 
  - Mobile: <768px
  - Tablet: 768px-1024px
  - Desktop: >1024px
- Container queries for component-level responsiveness
- Responsive grid: `grid-cols-1 md:grid-cols-2 lg:grid-cols-4`

### Browser Support
- Modern browsers with Tailwind CSS support
- Dark mode via `prefers-color-scheme` media query
- Flexbox and Grid layout support

### Accessibility
- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Color contrast compliance with WCAG standards
- Focus indicators for keyboard users

---

## Navigation Structure

All dashboards share a consistent navigation model:

### Left Sidebar
**Navigation Items** (across all dashboards):
1. **Territory Overview** - High-level geographic view
2. **Time Analysis** - Temporal data exploration
3. **Analytical Layers** - Multi-dimensional data analysis
4. **Census Data** - Demographic/infrastructure data

**Footer**:
- Documentation link
- Support link
- Apply Filters button

### Top Bar
- **Left**: UrbanMobility BI branding
- **Center**: Search box ("Search operational data...")
- **Right**: Notifications, Settings, User Profile

---

## Dashboard Workflow

### Common User Flows

**1. Data Exploration**
```
1. Select Territory/Filter from Sidebar
2. Browse Territory Overview map
3. Filter by Time period or analytical layer
4. Click into specific zone for details
5. View incident/sensor/metric details
6. Download or share view
```

**2. Alert Response**
```
1. View Active Alerts panel
2. Click on incident description
3. Review zone location and metrics
4. Trigger Strategic Action (Dispatch Crew, etc.)
5. Track resolution status
```

**3. Performance Monitoring**
```
1. View KPI dashboard
2. Scan trend indicators
3. Drill into underperforming zones
4. Analyze root causes via Analytical Layers
5. Generate reports for stakeholders
```

---

## Data Metrics & KPIs

### Universal Metrics

**Performance Indicators**:
- % Change (↑↓ with color coding)
- Trend sparklines
- Time period comparisons
- Geographic breakdowns

**Common Units**:
- **Capacity**: %, counts (e.g., 14,295 sensors)
- **Emissions**: kT (kilotons), g/km, %
- **Cost**: € (euros)
- **Quality**: Index (0-100), AQI scale
- **Density**: Count/km²
- **Efficiency**: kWh/km

### Territory-Based Segmentation
- Nantes Centre
- Strasbourg Europe
- Lyon Part-Dieu
- Paris La Défense
- Toulouse Capitole
- Nice Promenade
- Paris Region
- Aix-Marseille-Provence
- Lyon Métropole
- Lille Métropole
- Bordeaux Métropole

---

## Accessibility & Performance

### Accessibility Features
- Semantic HTML (`<nav>`, `<main>`, `<header>`, `<aside>`)
- ARIA attributes on custom components
- Keyboard navigation (Tab, Enter, Arrow keys)
- Focus management
- Color not the only indicator of status
- High contrast ratios (WCAG AA/AAA)
- Screen reader friendly labeling

### Performance Optimizations
- Tailwind CSS purge (production build)
- Lazy loading for charts and maps
- Responsive images
- Minimal re-renders in interactive components
- Efficient CSS with utility classes

---

## Development Guidelines

### Code Structure
Each dashboard (`code.html`) follows this structure:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <!-- Meta tags, viewport, fonts, icons -->
    <!-- Tailwind CSS CDN with plugins -->
    <!-- Custom styles and configurations -->
  </head>
  <body>
    <!-- SideNavBar -->
    <nav>...</nav>
    
    <!-- Main Content Area -->
    <div class="flex-1 ml-[280px]">
      <!-- TopNavBar -->
      <header>...</header>
      
      <!-- Canvas (Main Content) -->
      <main>...</main>
    </div>
  </body>
</html>
```

### Customization Points

**1. Colors**
- Update Tailwind config in `<script id="tailwind-config">`
- Modify `theme.extend.colors`
- All dashboards share the same palette

**2. Typography**
- Font family in Google Fonts link
- Font sizes in `theme.extend.fontSize`
- Line heights in type definitions

**3. Layout**
- Sidebar width: `w-[280px]` and `ml-[280px]`
- Grid columns: `grid-cols-1 md:grid-cols-2 lg:grid-cols-4`
- Spacing: Modify `theme.extend.spacing`

**4. Icons**
- Material Symbols from Google Fonts
- Change `data-icon` attribute
- Adjust size with `text-{size}` utilities

---

## File Organization Best Practices

### Adding New Dashboards
1. Create new folder in `stitch_urban_mobility_analytics/`
2. Add `code.html` file
3. Import Tailwind CSS with all plugins
4. Use shared color palette from Design system
5. Follow left-sidebar + main-content layout pattern
6. Include navigation items consistently

### Updating Styling
- Keep all dashboards using the same Tailwind config
- Don't override core colors without updating Design system
- Test dark mode across all dashboards
- Validate responsive breakpoints

### Maintenance
- Keep Design.md in sync with color/typography changes
- Document any new component patterns
- Test accessibility after modifications
- Validate performance on slow networks

---

## Summary

The `stitch_urban_mobility_analytics` folder provides a cohesive, enterprise-grade analytics platform for urban mobility management. It combines:

✅ **Consistent Design System** - Material Design 3 with custom urban mobility theme  
✅ **Multiple Perspectives** - Real-time, operational, strategic, and environmental views  
✅ **Enterprise Features** - Advanced filtering, dark mode, responsive design, accessibility  
✅ **Data Density** - High-information dashboards without cognitive overload  
✅ **Professional Aesthetic** - Authoritative yet progressive corporate modern style  
✅ **Accessibility** - WCAG compliance, keyboard navigation, screen reader support  

Perfect for government agencies, urban planners, and transit operators managing complex multi-dimensional urban mobility data in real-time.
