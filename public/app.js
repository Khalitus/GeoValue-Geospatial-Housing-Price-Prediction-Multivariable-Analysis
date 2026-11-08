/* GeoValue research interface: no training, invented metrics, or client-side price estimates. */
const $ = id => document.getElementById(id);
const state = {report:null, geoReport:null, points:null, surface:null, layer:'observed', drawn:new Set(), supported:[]};
const finite = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
const formatNumber = (value, decimals=0) => finite(value) ? Number(value).toLocaleString('en-US',{minimumFractionDigits:decimals,maximumFractionDigits:decimals}) : '—';
const usd = (value,decimals=0) => finite(value) ? '$'+formatNumber(value,decimals) : '—';
const plotConfig = {responsive:true,displayModeBar:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'png',filename:'geovalue-analysis'}};
const layoutBase = {paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Inter,Helvetica Neue,Arial,sans-serif',color:'#65736c',size:11},margin:{l:55,r:18,t:22,b:42},hoverlabel:{bgcolor:'#fff',font:{color:'#202925',size:12}}};
const sequential = [[0,'#e9f0e9'],[.25,'#b5d5c7'],[.55,'#64a18f'],[.8,'#347365'],[1,'#173e34']];
const errorScale = [[0,'#f7f2e6'],[.45,'#dbc299'],[1,'#986b35']];

async function requestJSON(url, options={}){
  let response;
  try {response=await fetch(url,{...options,headers:{'Accept':'application/json',...(options.headers||{})}})}
  catch {throw new Error('Unable to reach GeoValue. Check the server and try again.');}
  let body;
  try {body=await response.json()} catch {throw new Error('The server returned an unexpected response.');}
  if(!response.ok){
    let explanation=typeof body.detail==='string'?body.detail:'';
    if(!explanation && Array.isArray(body.detail)){
      explanation=body.detail.map(issue=>{
        const location=Array.isArray(issue.loc)?issue.loc.filter(x=>x!=='body').join(' › '):'Input';
        return `${location}: ${issue.msg || 'invalid value'}`;
      }).join('; ');
    }
    throw new Error(explanation || `Request failed (HTTP ${response.status}).`);
  }
  return body;
}
function errorMessage(id,message){const el=$(id);el.textContent=message;el.hidden=false;}
function clearError(id){$(id).textContent='';$(id).hidden=true;}
function emptyChart(id,message){const el=$(id);if(el){el.innerHTML='';const e=document.createElement('div');e.className='empty-plot';e.textContent=message;el.append(e);}}
function canPlot(id){if(window.Plotly)return true;emptyChart(id,'Charts could not load. Check your network connection and reload this page.');return false;}
function plotReact(id,data,layout){
  if(!canPlot(id))return Promise.resolve();
  try {
    const result=window.Plotly.react($(id),data,layout,plotConfig);
    state.drawn.add(id);
    return Promise.resolve(result).catch(()=>emptyChart(id,'Unable to render this chart. Try refreshing the page.'));
  } catch {
    emptyChart(id,'Unable to render this chart. Try refreshing the page.');
    return Promise.resolve();
  }
}
function setStatus(type,text){const el=$('model-status');el.className='service-status '+type;el.querySelector('.status-text').textContent=text;}

const pages=['overview','predict','geography','calculus','about'];
function navigate(page,{updateHash=true}={}){
  if(!pages.includes(page))page='overview';
  for(const element of document.querySelectorAll('.page'))element.classList.toggle('active',element.id==='page-'+page);
  for(const element of document.querySelectorAll('.nav-link')){
    const active=element.dataset.page===page;
    element.classList.toggle('active',active);
    if(active)element.setAttribute('aria-current','page');else element.removeAttribute('aria-current');
  }
  if(updateHash && location.hash!==`#${page}`)history.pushState(null,'',`#${page}`);
  window.scrollTo(0,0);
  requestAnimationFrame(()=>{
    if(page==='overview')renderOverview();
    if(page==='geography')renderGeography();
    if(page==='calculus')renderCalculus();
    for(const plot of document.querySelectorAll('.page.active .js-plotly-plot')){
      if(window.Plotly && window.Plotly.Plots)window.Plotly.Plots.resize(plot);
    }
  });
}

function renderOverview(){
  const r=state.report;
  if(!r){emptyChart('model-chart','Training report is unavailable. Run the deployment preparation script.');return;}
  const metrics=r.held_out_test || {};
  $('stat-r2').textContent=finite(metrics.r2)?formatNumber(metrics.r2,4):'—';
  $('stat-mae').textContent=usd(metrics.mae);
  $('stat-rmse').textContent=usd(metrics.rmse);
  const g=r.geographic_feature_experiment;
  if(g && g.with_geography && g.without_geography){
    const withGeo=g.with_geography.rmse, withoutGeo=g.without_geography.rmse;
    const reduction=withGeo>0 && withoutGeo>0?(withoutGeo-withGeo)/withoutGeo*100:0;
    $('stat-geo').textContent=formatNumber(reduction,1)+'%';
    $('impact-score').textContent=formatNumber(reduction,1)+'%';
    $('geo-with').textContent=usd(withGeo);
    $('geo-without').textContent=usd(withoutGeo);
    $('bar-with').style.width=(100*Math.min(1,withGeo/withoutGeo))+'%';
    $('bar-without').style.width='100%';
  }
  if(!r.validation_results)return;
  const models=[['Linear regression','linear'],['Random forest','random_forest'],['XGBoost','xgboost']];
  const rows=models.filter(([_,key])=>finite(r.validation_results[key]?.validation?.rmse));
  if(!rows.length){emptyChart('model-chart','Validation data is unavailable.');return;}
  const values=rows.map(([_,key])=>r.validation_results[key].validation.rmse);
  plotReact('model-chart',[{type:'bar',orientation:'h',x:values,y:rows.map(r=>r[0]),marker:{color:rows.map(row=>row[1]===r.selected_model?'#26584a':'#b9cdc0')},text:values.map(v=>usd(v)),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>RMSE %{x:$,.0f}<extra></extra>'}],{
    ...layoutBase,margin:{l:105,r:90,t:12,b:35},xaxis:{showgrid:true,gridcolor:'#edf0eb',zeroline:false,tickprefix:'$',tickformat:'~s'},yaxis:{autorange:'reversed',showgrid:false,automargin:true},showlegend:false
  });
}
function housingPoints(){return Array.isArray(state.points?.points)?state.points.points:[];}
function renderMap(){
  const rows=housingPoints();
  if(!rows.length){emptyChart('housing-map','No map sample is available. Run prepare_deployment.py to export the data.');return;}
  $('map-point-count').textContent=`${formatNumber(rows.length)} sampled census areas · select a point to use its coordinates`;
  const layer=state.layer;
  const vals=rows.map(row=>layer==='error'?Math.abs(row.predicted-row.observed):row[layer]);
  const text=layer==='error'?'Absolute error (1990 USD)':layer==='predicted'?'Predicted value (1990 USD)':'Observed value (1990 USD)';
  const trace={type:'scatter',mode:'markers',x:rows.map(row=>row.lon),y:rows.map(row=>row.lat),
    marker:{size:7,opacity:.76,color:vals,colorscale:layer==='error'?errorScale:sequential,showscale:true,colorbar:{title:{text:'1990 USD'},thickness:11,len:.72,outlinewidth:0},line:{width:0}},
    customdata:rows.map(row=>[row.observed,row.predicted,Math.abs(row.predicted-row.observed)]),
    hovertemplate:'Latitude %{y:.3f}°, longitude %{x:.3f}°<br>Observed: $%{customdata[0]:,.0f}<br>Predicted: $%{customdata[1]:,.0f}<br>Absolute error: $%{customdata[2]:,.0f}<extra></extra>'};
  plotReact('housing-map',[trace],{...layoutBase,margin:{l:54,r:70,t:15,b:48},
    xaxis:{title:'Longitude (°)',gridcolor:'#edf1ec',zeroline:false},
    yaxis:{title:'Latitude (°)',gridcolor:'#edf1ec',zeroline:false,scaleanchor:'x',scaleratio:1},showlegend:false}).then(()=>{
  const el=$('housing-map');
    if(typeof el.removeAllListeners==='function')el.removeAllListeners('plotly_click');
    if(typeof el.on==='function')el.on('plotly_click',event=>{
    const point=event?.points?.[0];
    if(!point || !finite(point.x) || !finite(point.y))return;
    const form=$('predict-form');
    form.elements.namedItem('longitude').value=Number(point.x).toFixed(5);
    form.elements.namedItem('latitude').value=Number(point.y).toFixed(5);
    markEstimateStale();navigate('predict');
    $('prediction-description').textContent='Location copied from the map. Adjust the census-area attributes and calculate a new estimate.';
    });
  });
}
function renderGeography(){
  renderMap();
  const s=state.surface;
  if(!s){emptyChart('surface-plot','Run prepare_deployment.py to export the modeled surface.');emptyChart('contour-plot','Run prepare_deployment.py to export the modeled surface.');return;}
  const axes={x:s.x_km,y:s.y_km,z:s.values,colorscale:sequential,showscale:true,hovertemplate:'Value: $%{z:,.0f}<extra></extra>'};
  const canvas=document.createElement('canvas');
  const hasWebGL=Boolean(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
  if(hasWebGL){
    plotReact('surface-plot',[{type:'surface',...axes,colorbar:{thickness:10,len:.7}}],{
      ...layoutBase,margin:{l:0,r:0,t:0,b:0},scene:{aspectmode:'manual',aspectratio:{x:1,y:1,z:.44},xaxis:{title:'Easting (km)',gridcolor:'#e9efeb'},yaxis:{title:'Northing (km)',gridcolor:'#e9efeb'},zaxis:{title:'USD',gridcolor:'#e9efeb'},camera:{eye:{x:1.45,y:1.6,z:.98}}}
    });
  } else {
    plotReact('surface-plot',[{type:'heatmap',...axes,colorscale:sequential,colorbar:{thickness:10,len:.7}}],{
      ...layoutBase,xaxis:{title:'Easting (km)'},yaxis:{title:'Northing (km)',scaleanchor:'x',scaleratio:1},
      annotations:[{xref:'paper',yref:'paper',x:.5,y:1.13,text:'3D requires WebGL · top-down view shown',showarrow:false,font:{size:10,color:'#68786e'}}]
    });
  }
  plotReact('contour-plot',[{type:'contour',...axes,contours:{showlabels:true,labelfont:{size:10}},colorbar:{thickness:9,len:.68}}],{
    ...layoutBase,xaxis:{title:'Easting (km)',gridcolor:'#eef1ed'},yaxis:{title:'Northing (km)',scaleanchor:'x',scaleratio:1,gridcolor:'#eef1ed'}
  });
}
function renderCalculus(){
  const report=state.geoReport;
  if(report){
    $('integral-area').textContent=formatNumber(report.numerical_double_integral?.valid_area_km2,0);
    $('integral-mean').textContent=usd(report.numerical_double_integral?.mean_surface_value_usd);
    $('mean-gradient').textContent=formatNumber(report.gradient_magnitude?.mean_usd_per_km,1);
  }
  const s=state.surface;
  if(!s){emptyChart('gradient-plot','Numerical gradient data is unavailable. Prepare the deployment assets.');return;}
  const magnitude=s.d_dx.map((row,j)=>row.map((gx,i)=>finite(gx)&&finite(s.d_dy[j][i])?Math.hypot(gx,s.d_dy[j][i]):null));
  const arrows=[];
  const step=Math.max(4,Math.ceil(s.x_km.length/11));
  const scale=(s.x_km[1]-s.x_km[0])*step*.55;
  for(let j=1;j<s.y_km.length-1;j+=step){for(let i=1;i<s.x_km.length-1;i+=step){
    const gx=s.d_dx[j][i],gy=s.d_dy[j][i];
    if(!finite(gx)||!finite(gy))continue;
    const m=Math.hypot(gx,gy);if(!m||!Number.isFinite(m))continue;
    const x=s.x_km[i],y=s.y_km[j];
    arrows.push({x:x+scale*gx/m,y:y+scale*gy/m,ax:x,ay:y,xref:'x',yref:'y',axref:'x',ayref:'y',showarrow:true,arrowhead:2,arrowsize:.7,arrowwidth:1.1,arrowcolor:'#183f35',opacity:.75});
  }}
  plotReact('gradient-plot',[{type:'heatmap',x:s.x_km,y:s.y_km,z:magnitude,colorscale:sequential,hovertemplate:'Gradient magnitude: %{z:,.0f} USD/km<extra></extra>',colorbar:{title:{text:'USD/km'},thickness:10,len:.68}}],{
    ...layoutBase,annotations:arrows,xaxis:{title:'Easting (km)',gridcolor:'#f0f2ed'},yaxis:{title:'Northing (km)',scaleanchor:'x',scaleratio:1,gridcolor:'#f0f2ed'}
  });
}

function formJSON(form){
  const raw=Object.fromEntries(new FormData(form).entries());
  for(const key in raw){if(key==='ocean_proximity')continue;raw[key]=raw[key]===''?null:Number(raw[key]);}
  return raw;
}
function markEstimateStale(){
  $('prediction-note').hidden=true;
  if($('prediction-result').dataset.computed==='true'){
    $('prediction-description').textContent='Inputs changed. Recalculate to refresh this estimate.';
  }
}
async function predict(event){
  event.preventDefault();clearError('predict-error');
  const submit=$('predict-submit');const original=submit.innerHTML;
  submit.disabled=true;submit.textContent='Calculating…';
  $('prediction-description').textContent='Calculating a model estimate…';
  try{
    const params=formJSON($('predict-form'));
    const result=await requestJSON('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    $('prediction-result').textContent=usd(result.predicted_median_house_value);
    $('prediction-result').dataset.computed='true';
    $('prediction-description').textContent=result.interpretation;
    const note=$('prediction-note');
    if(Array.isArray(result.warnings)&&result.warnings.length){note.textContent=result.warnings.join(' ');note.hidden=false;}
    else{note.hidden=true;note.textContent='';}
  }catch(error){
    errorMessage('predict-error',error.message);
    $('prediction-description').textContent='Could not calculate a prediction. Review the input values and try again.';
  }finally{submit.disabled=false;submit.innerHTML=original;}
}
function restoreExample(){
  const example={longitude:-122.23,latitude:37.88,housing_median_age:41,total_rooms:880,total_bedrooms:129,population:322,households:126,median_income:8.3252,ocean_proximity:'NEAR BAY'};
  for(const [name,value] of Object.entries(example))$('predict-form').elements.namedItem(name).value=value;
  markEstimateStale();clearError('predict-error');
}

function buildSupported(){
  const s=state.surface;if(!s)return;
  state.supported=[];
  for(let row=0;row<s.values.length;row++){
    for(let col=0;col<s.values[row].length;col++){
      if(!finite(s.values[row][col])||!finite(s.d_dx[row][col])||!finite(s.d_dy[row][col]))continue;
      const latitude=s.latitude[row][col],longitude=s.longitude[row][col];
      if(Number.isFinite(latitude)&&Number.isFinite(longitude))state.supported.push({latitude,longitude});
    }
  }
  const select=$('gradient-presets');
  select.querySelectorAll('option:not(:first-child)').forEach(el=>el.remove());
  const anchors=[['Bay Area',-122.3,37.75],['Sacramento region',-121.45,38.55],['Los Angeles region',-118.25,34.05],['San Diego region',-117.15,32.73]];
  for(const [name,longitude,latitude] of anchors){
    const match=nearestSupported(longitude,latitude);
    if(!match || match.distance>95)continue;
    const option=document.createElement('option');
    option.value=String(state.supported.indexOf(match.point));
    option.textContent=`${name} · ${formatNumber(match.point.latitude,2)}°, ${formatNumber(match.point.longitude,2)}°`;
    select.append(option);
  }
  if(select.options.length<=1){
    const point=state.supported[Math.floor(state.supported.length/2)];
    if(point){const option=document.createElement('option');option.value=String(state.supported.indexOf(point));option.textContent='Supported sample region';select.append(option);}
  }
  const match=nearestSupported(-122.23,37.88);
  if(match){$('gradient-form').elements.namedItem('latitude').value=match.point.latitude.toFixed(5);$('gradient-form').elements.namedItem('longitude').value=match.point.longitude.toFixed(5);}
  updateLocationGuidance();
}
function nearestSupported(longitude,latitude){
  if(!state.supported.length || !Number.isFinite(longitude) || !Number.isFinite(latitude))return null;
  let closest=null;let min=Infinity;
  const rad=Math.PI/180;
  for(const point of state.supported){
    const latDistance=(point.latitude-latitude)*111.2;
    const lonDistance=(point.longitude-longitude)*111.2*Math.cos(latitude*rad);
    const distance=Math.hypot(latDistance,lonDistance);
    if(distance<min){min=distance;closest=point;}
  }
  return {point:closest,distance:min};
}
function updateLocationGuidance(){
  const params=formJSON($('gradient-form'));
  const closest=nearestSupported(params.longitude,params.latitude);
  const label=$('location-guidance');
  if(!closest){label.textContent='A supported geographical surface is required for derivatives.';return;}
  if(closest.distance<=30){label.textContent=`Supported grid: nearest point ${formatNumber(closest.distance,1)} km away. Bearings use projected grid north.`;}
  else{label.textContent=`Outside the 30 km support limit (${formatNumber(closest.distance,1)} km away). Choose a preset or find the nearest supported point.`;}
}
function snapGradient(){
  const params=formJSON($('gradient-form'));
  const nearest=nearestSupported(params.longitude,params.latitude);
  if(!nearest){errorMessage('gradient-error','Surface data is unavailable.');return;}
  $('gradient-form').elements.namedItem('latitude').value=nearest.point.latitude.toFixed(6);
  $('gradient-form').elements.namedItem('longitude').value=nearest.point.longitude.toFixed(6);
  $('gradient-presets').value='';
  clearError('gradient-error');$('gradient-result').hidden=true;
  updateLocationGuidance();
}
async function gradient(event){
  event.preventDefault();clearError('gradient-error');$('gradient-result').hidden=true;
  const button=$('gradient-submit');const original=button.innerHTML;button.disabled=true;button.textContent='Calculating…';
  try{
    const params=formJSON($('gradient-form'));
    const body=await requestJSON('/api/gradient',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    const direction=body.directional_derivative_usd_per_km;
    $('direction-result').textContent=`${direction>0?'+':''}${formatNumber(direction,2)} USD/km`;
    $('direction-description').textContent=Math.abs(direction)<.005?'Approximately flat in the selected direction on the smoothed scenario surface.':direction>0?'Increasing in the selected direction on the smoothed scenario surface.':'Decreasing in the selected direction on the smoothed scenario surface.';
    $('magnitude-result').textContent=formatNumber(body.magnitude_usd_per_km,2)+' USD/km';
    $('ascent-result').textContent=formatNumber(body.steepest_ascent_bearing_grid_degrees,1)+'° (grid north)';
    $('distance-result').textContent=formatNumber(body.nearest_distance_km,2)+' km';
    $('gradient-result').hidden=false;
  }catch(error){errorMessage('gradient-error',error.message.includes('No supported geographical gradient')?'That location is too far from a valid grid point. Choose a region above or use “Find nearest supported point”.':error.message);}
  finally{button.disabled=false;button.innerHTML=original;}
}
async function initialise(){
  document.querySelectorAll('[data-page]').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.page)));
  document.querySelectorAll('[data-goto]').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.goto)));
  window.addEventListener('popstate',()=>navigate(location.hash.slice(1),{updateHash:false}));
  window.addEventListener('hashchange',()=>navigate(location.hash.slice(1),{updateHash:false}));
  document.querySelectorAll('[data-layer]').forEach(button=>button.addEventListener('click',()=>{
    state.layer=button.dataset.layer;
    for(const element of document.querySelectorAll('[data-layer]')){
      const active=element===button;element.classList.toggle('active',active);element.setAttribute('aria-pressed',String(active));
    }
    renderMap();
  }));
  $('predict-form').addEventListener('submit',predict);
  $('predict-form').addEventListener('input',markEstimateStale);
  $('restore-example').addEventListener('click',restoreExample);
  $('gradient-form').addEventListener('submit',gradient);
  $('snap-gradient').addEventListener('click',snapGradient);
  $('gradient-form').addEventListener('input',()=>{updateLocationGuidance();$('gradient-result').hidden=true;});
  $('gradient-presets').addEventListener('change',event=>{
    const index=Number(event.target.value);
    if(event.target.value==='' || !state.supported[index])return;
    const point=state.supported[index];
    $('gradient-form').elements.namedItem('longitude').value=point.longitude.toFixed(6);
    $('gradient-form').elements.namedItem('latitude').value=point.latitude.toFixed(6);
    clearError('gradient-error');$('gradient-result').hidden=true;updateLocationGuidance();
  });
  $('bearing').addEventListener('input',()=>{$('bearing-value').textContent=$('bearing').value+'° clockwise from projected grid north';});
  const entries=[['report','/api/report'],['geoReport','/api/geo-report'],['points','/data/points.json'],['surface','/data/surface.json']];
  await Promise.all([
    ...entries.map(async ([key,url])=>{try{state[key]=await requestJSON(url);}catch{state[key]=null;}}),
    (async()=>{try{const health=await requestJSON('/api/health');setStatus(health.model_available?'ready':'unavailable',health.model_available?'Model ready':'Model missing');}
      catch {setStatus('unavailable','Service offline');}})()
  ]);
  if(state.surface)buildSupported();
  navigate(location.hash.slice(1)||'overview',{updateHash:false});
}
initialise();
