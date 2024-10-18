//Make Sure to import all the polygon of coordinates you want to get NDVI data for and store them as variables

// Cloud masking function
function cloudMask(image) {
  var scl = image.select('SCL');
  var mask = scl.eq(3).or(scl.gte(7).and(scl.lte(10)));
  return image.updateMask(mask.eq(0));
}

// Water masking function
function waterMask(image) {
  var scl = image.select('SCL');
  var mask = scl.eq(6);
  return image.updateMask(mask.eq(0));
}

// Add NDVI function
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
  return image.addBands(ndvi);
}

// Create reduce region function
function createReduceRegionFunction(geometry, reducer, scale, crs, bestEffort, maxPixels, tileScale) {
  return function(img) {
    var stat = img.reduceRegion({
      reducer: reducer,
      geometry: geometry,
      scale: scale,
      crs: crs,
      bestEffort: bestEffort,
      maxPixels: maxPixels,
      tileScale: tileScale
    });
    var properties = ee.Feature(null, stat).toDictionary();
    return ee.Feature(geometry, properties).set({millis: img.date().millis()});
  };
}

// Get NDVI images function
function getNDVIImages(aoi, startDate, endDate, CC) {
  var s2Polygon = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CC))
    .map(cloudMask)
    .map(waterMask)
    .map(addNDVI)
    .select('ndvi');

  var reduceNDVI = createReduceRegionFunction(
    aoi, ee.Reducer.median(), 10, 'EPSG:4326', true, 1e13, 4
  );

  var ndviStatFc = ee.FeatureCollection(s2Polygon.map(reduceNDVI));
  
  return ndviStatFc;
}

// Main processing
var startDate = '2020-06-15';
var endDate = '2020-06-29';
var CC = 60
var emptyFc = ee.FeatureCollection([]);

// Process each field
var processField = function(feature) {
  var farmId = feature.get('Field_ID');
  var geometry = feature.geometry();
  
  var statFc = getNDVIImages(geometry, startDate, endDate);
  return statFc.map(function(f) {
    return f.set('Field_ID', farmId);
  });
};

var allFields = field_2020.map(processField).flatten();

// Export the results
Export.table.toAsset({
  collection: allFields,
  description: 'ndvi_stat_fc_asset_export',
  assetId: 'projects/ee-nickodg123/assets/ndvi_2020_inteval_1'
});
