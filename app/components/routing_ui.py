"""Streamlit UI Component for Hospital Selection, Routing & Interactive Map.

Renders deterministic hospital suitability rankings, road-network ambulance routes,
interactive Folium map visualizations, and external Google Maps links.
"""

from typing import Dict, Any, Optional
import streamlit as st
import folium
from streamlit_folium import st_folium

from core.sos.location import SOSLocation
from core.hospitals.ranker import HospitalRanker


@st.cache_resource
def get_hospital_ranker() -> HospitalRanker:
    return HospitalRanker()


def render_hospital_routing_tab(
    sos_location: SOSLocation,
    urgency_level: str = "HIGH",
):
    """Render the Hospital Intelligence & Ambulance Routing tab."""
    st.subheader("🚑 Nearest Suitable Hospital Selection & Route")
    st.markdown(
        "Hospitals are ranked primarily by **documented suitability** and secondarily by **road network travel time**. "
        "Routes strictly originate from the **accident SOS coordinates**."
    )

    if not sos_location or not sos_location.latitude or not sos_location.longitude:
        st.warning("Please provide a validated incident SOS location to compute emergency hospital routing.")
        return None

    ranker = get_hospital_ranker()
    ranking_data = ranker.rank_hospitals(sos_location, urgency_level=urgency_level)

    if "error" in ranking_data and ranking_data.get("primary_recommendation") is None:
        st.error(ranking_data["error"])
        return None

    primary = ranking_data["primary_recommendation"]
    alternatives = ranking_data["alternatives"]

    # 1. Primary Recommendation Banner
    fac = primary["facility"]
    route = primary["route"]

    st.success(f"### 🏥 Primary Recommended Facility: {fac['name']}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Road Distance", f"{route['distance_km']} km")
    p2.metric("Estimated Journey Time", f"~{route['estimated_time_minutes']} min")
    p3.metric("Facility Category", fac["category"].title())
    p4.metric("Emergency Tag", "YES (24/7)" if fac["has_documented_emergency"] else "Undocumented")

    st.markdown(f"**Address**: {fac['address']}")
    if fac.get("phone"):
        st.markdown(f"**Emergency Phone**: `{fac['phone']}` *(Prototype note: App does not place calls)*")
    else:
        st.caption("Contact phone: Undocumented in OpenStreetMap cache.")

    st.markdown(f"**Ranking Rationale**: {primary['ranking_reason']}")
    st.markdown(f"🔗 [Open Navigation in Google Maps]({route['google_maps_url']})")

    # 2. Interactive Route Map using Folium
    st.markdown("#### Interactive Ambulance Response Route")
    start_lat = sos_location.latitude
    start_lon = sos_location.longitude
    dest_lat = fac["latitude"]
    dest_lon = fac["longitude"]

    # Center map midway
    mid_lat = (start_lat + dest_lat) / 2.0
    mid_lon = (start_lon + dest_lon) / 2.0

    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13, tiles="OpenStreetMap")

    # Incident Marker (Red)
    folium.Marker(
        [start_lat, start_lon],
        popup=f"🚨 Incident SOS: {sos_location.location_name}",
        tooltip="Incident Location (SOS Origin)",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(m)

    # Primary Hospital Marker (Green)
    folium.Marker(
        [dest_lat, dest_lon],
        popup=f"🏥 Recommended: {fac['name']}<br>Est. Time: {route['estimated_time_minutes']} min",
        tooltip="Primary Hospital",
        icon=folium.Icon(color="green", icon="plus"),
    ).add_to(m)

    # Route Polyline (Blue)
    geometry = route.get("route_geometry", [])
    if geometry:
        folium.PolyLine(
            locations=geometry,
            color="#2A81CB",
            weight=5,
            opacity=0.85,
            tooltip=f"Calculated Road Route: {route['distance_km']} km",
        ).add_to(m)

    # Alternative Hospital Markers (Orange)
    for alt in alternatives:
        afac = alt["facility"]
        aroute = alt["route"]
        folium.Marker(
            [afac["latitude"], afac["longitude"]],
            popup=f"🏥 Alt: {afac['name']}<br>Road Dist: {aroute['distance_km']} km",
            tooltip=f"Alternative: {afac['name']}",
            icon=folium.Icon(color="orange", icon="info-sign"),
        ).add_to(m)

    # Render folium map
    st_folium(m, width=900, height=450)

    # 3. Alternative Facilities Table
    if alternatives:
        st.markdown("#### Alternative Eligible Medical Facilities")
        alt_rows = []
        for a in alternatives:
            afac = a["facility"]
            aroute = a["route"]
            alt_rows.append({
                "Facility Name": afac["name"],
                "Category": afac["category"].title(),
                "Documented Emergency": "YES" if afac["has_documented_emergency"] else "Undocumented",
                "Road Distance (km)": aroute["distance_km"],
                "Est. Time (min)": aroute["estimated_time_minutes"],
                "Ranking Reason": a["ranking_reason"],
            })
        st.table(alt_rows)

    st.info(
        "⚠️ **Safety Disclosure**: Ambulance fleet, routing replay, and hospital recommendations are simulated for demonstration. "
        "No real emergency vehicle has been contacted, and hospital admission is not verified."
    )

    return ranking_data
