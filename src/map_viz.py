"""Interactive map visualization of AADBT predictions."""

import logging
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium import plugins

logger = logging.getLogger(__name__)


def compute_aadbt_by_site(
    df: pd.DataFrame, count_col: str = "count"
) -> pd.DataFrame:
    """
    Compute Average Annual Daily Bike Traffic (AADBT) per site.
    
    AADBT = average daily total across all days in the period.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with site_id, date, count columns
    count_col : str
        Column name for count
        
    Returns
    -------
    aadbt_df : DataFrame
        DataFrame with site_id, aadbt columns
    """
    # Sum hourly counts to daily
    daily_df = df.groupby(["site_id", "date"], as_index=False)[count_col].sum()
    
    # Average daily count per site
    aadbt_df = daily_df.groupby("site_id", as_index=False)[count_col].mean()
    aadbt_df = aadbt_df.rename(columns={count_col: "aadbt"})
    
    return aadbt_df


def create_aadbt_map(
    sites_df: pd.DataFrame,
    output_path: Path,
    zoom_start: int = 12,
) -> None:
    """
    Create interactive Folium map with AADBT visualizations.
    
    Parameters
    ----------
    sites_df : DataFrame
        DataFrame with columns: site_id, lat, lon, 
        aadbt_test_obs, aadbt_test_pred, aadbt_full_obs, aadbt_full_pred, error_pct
    output_path : Path
        Output HTML file path
    zoom_start : int
        Initial zoom level
    """
    logger.info("Creating interactive AADBT map...")
    
    # Compute center
    center_lat = sites_df["lat"].mean()
    center_lon = sites_df["lon"].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )
    
    # Normalize sizes for visualization
    max_aadbt_obs = sites_df["aadbt_test_obs"].max()
    max_aadbt_pred = sites_df["aadbt_test_pred"].max()
    max_aadbt_overall = max(max_aadbt_obs, max_aadbt_pred)
    
    # Create feature groups for layer control
    layer_obs = folium.FeatureGroup(name="Observed AADBT (test period)")
    layer_pred = folium.FeatureGroup(name="Predicted AADBT (test period)")
    
    # Add markers for each site
    for idx, row in sites_df.iterrows():
        # Popup content
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Site:</b> {row['site_id']}<br>
            <b>Location:</b> {row['lat']:.6f}, {row['lon']:.6f}<br>
            <hr>
            <b>Test Period:</b><br>
            &nbsp;&nbsp;Observed AADBT: {row['aadbt_test_obs']:.1f}<br>
            &nbsp;&nbsp;Predicted AADBT: {row['aadbt_test_pred']:.1f}<br>
            &nbsp;&nbsp;Error: {row['error_pct']:.1f}%<br>
            <hr>
            <b>Full Period:</b><br>
            &nbsp;&nbsp;Observed AADBT: {row['aadbt_full_obs']:.1f}<br>
            &nbsp;&nbsp;Predicted AADBT: {row['aadbt_full_pred']:.1f}<br>
        </div>
        """
        
        # Color by error
        error = abs(row["error_pct"])
        if error < 10:
            color_obs = "green"
            color_pred = "darkgreen"
        elif error < 25:
            color_obs = "orange"
            color_pred = "darkorange"
        else:
            color_obs = "red"
            color_pred = "darkred"
        
        # Observed layer
        radius_obs = 5 + (row["aadbt_test_obs"] / max_aadbt_overall) * 20
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius_obs,
            popup=folium.Popup(popup_html, max_width=300),
            color=color_obs,
            fill=True,
            fillColor=color_obs,
            fillOpacity=0.6,
            weight=2,
        ).add_to(layer_obs)
        
        # Predicted layer
        radius_pred = 5 + (row["aadbt_test_pred"] / max_aadbt_overall) * 20
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius_pred,
            popup=folium.Popup(popup_html, max_width=300),
            color=color_pred,
            fill=True,
            fillColor=color_pred,
            fillOpacity=0.6,
            weight=2,
        ).add_to(layer_pred)
    
    # Add layers to map
    layer_obs.add_to(m)
    layer_pred.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
        <b>Error Legend</b><br>
        <i class="fa fa-circle" style="color:green"></i> < 10% error<br>
        <i class="fa fa-circle" style="color:orange"></i> 10-25% error<br>
        <i class="fa fa-circle" style="color:red"></i> > 25% error<br>
        <hr>
        <b>Circle size</b> ∝ AADBT
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(str(output_path))
    logger.info(f"Saved: {output_path}")


def map_viz_pipeline(artifacts_dir: Path) -> None:
    """
    Full map visualization pipeline.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with test_predictions.parquet, train.parquet, test.parquet
    """
    logger.info("Loading data...")
    
    train_df = pd.read_parquet(artifacts_dir / "features.parquet")
    test_df = pd.read_parquet(artifacts_dir / "test_predictions.parquet")
    
    # Ensure date column exists
    if "date" not in train_df.columns and "timestamp" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["timestamp"]).dt.date
    if "date" not in test_df.columns and "timestamp" in test_df.columns:
        test_df["date"] = pd.to_datetime(test_df["timestamp"]).dt.date
    
    logger.info(f"Train: {train_df.shape[0]:,} rows, {train_df['site_id'].nunique()} unique sites")
    logger.info(f"Test:  {test_df.shape[0]:,} rows, {test_df['site_id'].nunique()} unique sites")
    
    # Check if some sites are missing in test
    train_sites = set(train_df['site_id'].unique())
    test_sites = set(test_df['site_id'].unique())
    missing_in_test = train_sites - test_sites
    if missing_in_test:
        logger.warning(f"Sites present in train but not in test: {missing_in_test}")
    
    # Compute AADBT for test period (observed)
    logger.info("\nComputing observed AADBT for test period...")
    aadbt_test_obs = compute_aadbt_by_site(test_df, count_col="count")
    aadbt_test_obs = aadbt_test_obs.rename(columns={"aadbt": "aadbt_test_obs"})
    
    # Compute AADBT for test period (predicted)
    logger.info("Computing predicted AADBT for test period...")
    aadbt_test_pred = compute_aadbt_by_site(test_df, count_col="predicted_count")
    aadbt_test_pred = aadbt_test_pred.rename(columns={"aadbt": "aadbt_test_pred"})
    
    # Compute AADBT for full period (observed)
    logger.info("Computing observed AADBT for full period...")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    aadbt_full_obs = compute_aadbt_by_site(full_df, count_col="count")
    aadbt_full_obs = aadbt_full_obs.rename(columns={"aadbt": "aadbt_full_obs"})
    
    # For full predicted, approximate using test predictions ratio
    # (since we don't have train predictions, use test as proxy)
    logger.info("Computing predicted AADBT for full period (approximated)...")
    test_ratio = aadbt_test_pred.merge(aadbt_test_obs, on="site_id")
    test_ratio["pred_ratio"] = (
        test_ratio["aadbt_test_pred"] / test_ratio["aadbt_test_obs"]
    )
    
    aadbt_full_pred = aadbt_full_obs.merge(
        test_ratio[["site_id", "pred_ratio"]], on="site_id", how="left"
    )
    aadbt_full_pred["pred_ratio"] = aadbt_full_pred["pred_ratio"].fillna(1.0)
    aadbt_full_pred["aadbt_full_pred"] = (
        aadbt_full_pred["aadbt_full_obs"] * aadbt_full_pred["pred_ratio"]
    )
    aadbt_full_pred = aadbt_full_pred[["site_id", "aadbt_full_pred"]]
    
    # Merge all together
    logger.info("Merging AADBT metrics by site...")
    sites_aadbt = (
        aadbt_test_obs.merge(aadbt_test_pred, on="site_id")
        .merge(aadbt_full_obs, on="site_id")
        .merge(aadbt_full_pred, on="site_id")
    )
    
    # Compute error percentage (handle division by zero)
    sites_aadbt["error_pct"] = np.where(
        sites_aadbt["aadbt_test_obs"] > 0,
        100 * (sites_aadbt["aadbt_test_pred"] - sites_aadbt["aadbt_test_obs"]) / sites_aadbt["aadbt_test_obs"],
        np.nan  # Set to NaN if observed is 0 (no data for this site in test)
    )
    
    # Add coordinates (from test_df)
    site_coords = test_df[["site_id", "lat", "lon"]].drop_duplicates()
    sites_aadbt = sites_aadbt.merge(site_coords, on="site_id")
    
    logger.info(f"Total sites: {len(sites_aadbt)}")
    logger.info(f"\nAADBT Statistics (test period):")
    logger.info(f"  Observed: mean={sites_aadbt['aadbt_test_obs'].mean():.1f}, "
                f"std={sites_aadbt['aadbt_test_obs'].std():.1f}")
    logger.info(f"  Predicted: mean={sites_aadbt['aadbt_test_pred'].mean():.1f}, "
                f"std={sites_aadbt['aadbt_test_pred'].std():.1f}")
    logger.info(f"  Error: mean={sites_aadbt['error_pct'].mean():.1f}%, "
                f"std={sites_aadbt['error_pct'].std():.1f}%")
    
    # Create map
    create_aadbt_map(sites_aadbt, artifacts_dir / "cph_aadbt_map.html")
    
    # Save AADBT table
    sites_aadbt.to_csv(artifacts_dir / "aadbt_by_site.csv", index=False)
    logger.info(f"Saved: {artifacts_dir / 'aadbt_by_site.csv'}")
    
    logger.info("\nMap visualization complete!")

