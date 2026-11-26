"""Hospital recommendation and road route visualization."""
from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from core.sos.location import SOSLocation
from core.hospitals.ranker import HospitalRanker


@st.cache_resource
def get_hospital_ranker() -> HospitalRanker:
    return HospitalRanker()


def render_hospital_routing_tab(
    sos_location: SOSLocation | None,
    urgency_level: str = "HIGH",
    incident_text: str = "",
):
    """Render the nearest source-tagged hospital and its saved-graph road route."""
    st.subheader("Nearest suitable hospital and ambulance route")
    st.caption(
        "Facilities are filtered to source-tagged hospitals only. Ranking uses "
        "source-record hints and route time calculated on the saved road graph."
    )
    if sos_location is None:
        st.warning("Provide an SOS event, capture browser GPS at the incident, or verify operator coordinates.")
        return None
    if sos_location.is_stale:
        st.error("The SOS coordinates are stale. Refresh or verify the incident location before routing.")
        return None

    ranker = get_hospital_ranker()
    ranking = ranker.rank_hospitals(
        sos_location,
        urgency_level=urgency_level,
        incident_text=incident_text,
    )
    if ranking.get("error") or not ranking.get("primary_recommendation"):
        st.error(ranking.get("error", "No routable hospital found for this incident location."))
        st.info(
            "No straight-line distance has been substituted for a driving route. "
            "Try a verified location inside the saved Phagwara road-network coverage."
        )
        return ranking

    primary = ranking["primary_recommendation"]
    facility = primary["facility"]
    route = primary["route"]
    st.success(f"### Primary source-tagged hospital: {facility['name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calculated road distance", f"{route['distance_km']:.2f} km")
    c2.metric("Static estimated travel time", f"~{route['estimated_time_minutes']:.1f} min")
    c3.metric(
        "Emergency tag",
        "YES (details unverified)" if facility["has_documented_emergency"] else "Undocumented",
    )
    st.markdown(f"**Address:** {facility['address']}")
    st.markdown(f"**Facility category:** {facility['category'].title()}")
    st.markdown(
        f"**Specialty tag/name hint (not clinically verified):** "
        f"{facility['speciality'] or 'Not listed'}"
    )
    st.markdown(f"**Contact:** {facility['phone'] or 'Not listed in OSM'}")
    st.markdown(f"**Ranking rationale:** {primary['ranking_reason']}")
    if facility.get("source_url"):
        st.markdown(f"**Source record:** [OpenStreetMap hospital record]({facility['source_url']})")
    st.caption(
        f"Hospital snapshot freshness: {facility.get('data_freshness') or 'Not supplied by source'}. "
        "OSM emergency=yes does not establish 24/7 hours, staffing, trauma capability, "
        "current readiness, bed availability or admission acceptance."
    )

    midpoint = route["route_geometry"][len(route["route_geometry"]) // 2]
    route_map = folium.Map(location=midpoint, zoom_start=13, control_scale=True)
    folium.Marker(
        [sos_location.latitude, sos_location.longitude],
        tooltip="SOS accident origin",
        popup=f"Incident origin<br>Source: {sos_location.source}<br>{sos_location.location_name}",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(route_map)
    folium.Marker(
        [facility["latitude"], facility["longitude"]],
        tooltip=facility["name"],
        popup=f"{facility['name']}<br>{facility['address']}",
        icon=folium.Icon(color="green", icon="plus-sign"),
    ).add_to(route_map)
    folium.PolyLine(
        route["route_geometry"],
        color="#1d4ed8",
        weight=5,
        opacity=0.9,
        tooltip="Path on saved directed OpenStreetMap road graph",
    ).add_to(route_map)

    for alternative in ranking["alternatives"]:
        alt = alternative["facility"]
        folium.CircleMarker(
            [alt["latitude"], alt["longitude"]],
            radius=5,
            tooltip=f"Alternative: {alt['name']}",
            color="#f59e0b",
            fill=True,
        ).add_to(route_map)
    st_folium(route_map, height=480, use_container_width=True, returned_objects=[])
    st.markdown(f"[Open driving directions in Google Maps]({route['google_maps_url']})")
    st.caption(f"© OpenStreetMap contributors. {route['travel_time_note']}")

    if ranking["alternatives"]:
        st.markdown("#### Alternative routable hospitals")
        st.table([
            {
                "Hospital": item["facility"]["name"],
                "Category": item["facility"]["category"],
                "Road km": item["route"]["distance_km"],
                "Estimated min": item["route"]["estimated_time_minutes"],
                "Emergency tag": (
                    "yes (details unverified)"
                    if item["facility"]["has_documented_emergency"] else "undocumented"
                ),
                "Ranking explanation": item["ranking_reason"],
            }
            for item in ranking["alternatives"]
        ])

    st.info(
        "Recommendation is based on the available OSM record and static road graph. "
        "Clinical suitability, hospital readiness, opening status and acceptance must be confirmed by a human."
    )
    handoff = st.checkbox(
        "I have reviewed the SOS coordinates, hospital source and route",
        key="route-human-review",
    )
    if st.button("Prepare simulated ambulance handoff", disabled=not handoff):
        st.warning(
            "SIMULATION ONLY — route details are prepared for human review. "
            "No 112/108 call was made, no ambulance was contacted or assigned, "
            "and the hospital has not accepted a patient."
        )
    return ranking
