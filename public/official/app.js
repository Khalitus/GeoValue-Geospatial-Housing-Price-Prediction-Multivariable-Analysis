/* GeoValue official-data local dashboard. External requests are intentionally absent. */
(() => {
  'use strict';
  const data = {report:null, geo:null, reliability:null, points:null, surface:null};
  const rendered = new Set();
  let layer = 'observed';
  const $ = (id) => document.getElementById(id);
  const money = (value, digits=0) => Number.isFinite(Number(value)) ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:digits}).format(Number(value)) : '—';
  const number = (value, digits=0) => Number.isFinite(Number(value)) ? new Intl.NumberFormat('en-US',{maximumFractionDigits:digits}).format(Number(value)) : '—';
  const mark = (id,value) => {const node=$(id);if(node) node.textContent=value;};
  const finite = (value) => value !== null && Number.isFinite(Number(value));
  const config = {responsive:true,displayModeBar:false};
  const colors = {ink:'#26393b',sub:'#657779',accent:'#286d62',secondary:'#9bbcb3',lines:'#e2ebe7'};
  const plotBase = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Arial, sans-serif',color:colors.sub,size:11},margin:{l:68,r:20,t:23,b:55},showlegend:false};
  async function getJSON(url) {
    const response = await fetch(url,{cache:'no-store'});
    if(!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {const parsed=await response.json();if(parsed.detail)message=typeof parsed.detail==='string'?parsed.detail:JSON.stringify(parsed.detail);}catch (_){}
      throw new Error(message);
    }
    return response.json();
  }
  function showUnavailable(id,message) {const node=$(id);if(node)node.replaceChildren(Object.assign(document.createElement('div'),{className:'status-placeholder',textContent:message}));}
  function hasPlotly() {return !!window.Plotly;}
  function plot(id,traces,layout) {if(!hasPlotly()){showUnavailable(id,'Charts unavailable. The local Plotly bundle could not be loaded.');return Promise.resolve();}return window.Plotly.newPlot($(id),traces,{...plotBase,...layout},config).catch(()=>showUnavailable(id,'This chart could not be displayed on this device.'));}
  function setPage(page) {
    if(!$(`page-${page}`)) return;
    document.querySelectorAll('.page').forEach(s=>s.classList.toggle('active',s.id===`page-${page}`));
    document.querySelectorAll('.nav-item').forEach(btn=>{const selected=btn.dataset.page===page;btn.classList.toggle('selected',selected); if(selected)btn.setAttribute('aria-current','page');else btn.removeAttribute('aria-current');});
    if(window.location.hash!==`#${page}`)history.replaceState(null,'',`#${page}`);
    window.scrollTo(0,0);
    showCharts(page);
  }
  document.querySelectorAll('[data-page],[data-goto]').forEach(button=>button.addEventListener('click',()=>setPage(button.dataset.page||button.dataset.goto)));
  window.addEventListener('hashchange',()=>setPage(window.location.hash.slice(1)||'overview'));
  function drawOverview(){
    const r=data.report;
    if(!r) {showUnavailable('model-chart','Official training report unavailable. Train the county model first.');showUnavailable('geography-chart','Geographical experiment unavailable.');return;}
    mark('model-name',`${String(r.model_selected).replaceAll('_',' ')} · 2024 validation selection`);
    const test=r.held_out_2025_test || {};
    mark('stat-r2',number(test.r2,4));mark('stat-mae',money(test.mae));mark('stat-rmse',money(test.rmse));
    mark('stat-count',data.geo?.records!=null?number(data.geo.records):'—');
    const models=Object.entries(r.model_metrics||{}).filter(([,v])=>finite(v?.validation?.rmse));
    if(models.length)plot('model-chart',[{type:'bar',x:models.map(v=>v[0].replaceAll('_',' ')),y:models.map(v=>v[1].validation.rmse),marker:{color:models.map(v=>v[0]===r.model_selected?colors.accent:'#9cafab')},hovertemplate:'%{x}<br>%{y:$,.0f} RMSE<extra></extra>'}],{yaxis:{type:'log',title:'RMSE, USD · log scale',gridcolor:colors.lines,zeroline:false},xaxis:{tickangle:0}});
    const geo=r.geographical_ablation;
    if(geo)plot('geography-chart',[{type:'bar',orientation:'h',y:['Without location','With location'],x:[geo.validation_without_geo.rmse,geo.validation_with_geo.rmse],marker:{color:['#9cafab',colors.accent]},hovertemplate:'%{y}<br>%{x:$,.0f} RMSE<extra></extra>'}],{xaxis:{title:'Validation RMSE, USD',gridcolor:colors.lines},yaxis:{automargin:true},margin:{l:125,r:25,t:24,b:65}});
  }
  function drawMap(){
    const d=data.points;
    if(!d){showUnavailable('map-plot','Run geovalue-official analyze to prepare the 2025 map sample.');return;}
    mark('map-meta',`${number(d.count)} sampled of ${number(d.source_total)} held-out 2025 sales`);
    const values=d[layer];
    const title=layer==='absolute_error'?'Absolute error':layer==='observed'?'Observed price':'Predicted price';
    const trace={type:'scatter',mode:'markers',x:d.longitude,y:d.latitude,
      customdata:d.observed.map((val,i)=>[val,d.predicted[i],d.absolute_error[i]]),
      marker:{size:7,opacity:.78,color:values,colorscale:'Viridis',showscale:true,colorbar:{title:{text:'USD'},thickness:12,len:.7}},
      hovertemplate:`Longitude %{x:.4f}<br>Latitude %{y:.4f}<br>${title}: %{marker.color:$,.0f}<br>Observed %{customdata[0]:$,.0f}<br>Predicted %{customdata[1]:$,.0f}<br>Absolute error %{customdata[2]:$,.0f}<extra></extra>`};
    plot('map-plot',[trace],{xaxis:{title:'Longitude (°)',gridcolor:colors.lines},yaxis:{title:'Latitude (°)',gridcolor:colors.lines},margin:{l:65,r:40,t:22,b:55}}).then(()=>{
      const target=$('map-plot');if(target.removeAllListeners)target.removeAllListeners('plotly_click');
      target.on?.('plotly_click',event=>{
      const point=event.points?.[0];if(!point)return;
      const lon=Number(point.x),lat=Number(point.y);
      for(const id of ['property-form','gradient-form']){
        const form=$(id);if(form){form.elements.longitude.value=lon.toFixed(6);form.elements.latitude.value=lat.toFixed(6);}
      }
      mark('map-message','Location copied into the estimate and calculus forms. Property characteristics remain unchanged.');
      });
    });
  }
  function surfaceData(){const d=data.surface;if(!d) return null;return {x:d.x_km,y:d.y_km,z:d.values};}
  function drawGeography(){
    drawMap();const grid=surfaceData();
    if(!grid){showUnavailable('surface-plot','Official scenario surface not generated yet.');showUnavailable('contour-plot','No surface available.');return;}
    const contour={type:'contour',x:grid.x,y:grid.y,z:grid.z,colorscale:'Viridis',connectgaps:false,colorbar:{title:'USD'},contours:{showlabels:true},hovertemplate:'E %{x:.1f} km<br>N %{y:.1f} km<br>%{z:$,.0f}<extra></extra>'};
    const three={type:'surface',x:grid.x,y:grid.y,z:grid.z,colorscale:'Viridis',colorbar:{title:'USD'},connectgaps:false,hovertemplate:'E %{x:.1f} km<br>N %{y:.1f} km<br>Value %{z:$,.0f}<extra></extra>'};
    const canvas=document.createElement('canvas');const gl=!!(canvas.getContext('webgl')||canvas.getContext('experimental-webgl'));
    if(gl)plot('surface-plot',[three],{scene:{xaxis:{title:'Easting (km)'},yaxis:{title:'Northing (km)'},zaxis:{title:'Sale scenario USD'}},margin:{l:15,r:15,t:15,b:10}});
    else plot('surface-plot',[contour],{xaxis:{title:'Easting (km)'},yaxis:{title:'Northing (km)'},margin:{l:65,r:20,t:22,b:60}});
    plot('contour-plot',[contour],{xaxis:{title:'Easting (km)'},yaxis:{title:'Northing (km)',scaleanchor:'x',scaleratio:1},margin:{l:65,r:20,t:20,b:55}});
  }
  function drawCalculus(){
    const r=data.geo;
    if(r){const integral=r.numerical_double_integral||{};mark('stat-area',number(integral.valid_area_km2));mark('stat-surface-mean',money(integral.mean_surface_value_usd));mark('stat-gradient',number(r.mean_gradient_usd_per_km));}
    const s=data.surface;
    if(!s){showUnavailable('gradient-plot','Run geovalue-official analyze to enable the calculus explorer.');return;}
    const magnitude=s.values.map((row,i)=>row.map((v,j)=>finite(v)&&finite(s.d_dx[i][j])&&finite(s.d_dy[i][j])?Math.hypot(s.d_dx[i][j],s.d_dy[i][j]):null));
    const heat={type:'heatmap',x:s.x_km,y:s.y_km,z:magnitude,colorscale:'Blues',colorbar:{title:'USD/km'},hovertemplate:'E %{x:.1f} km<br>N %{y:.1f} km<br>Gradient %{z:,.0f} USD/km<extra></extra>'};
    const arrowsX=[],arrowsY=[];
    const step=Math.max(3,Math.floor(s.x_km.length/13));const gridspacing=(s.x_km[1]-s.x_km[0])*step*0.6;
    for(let i=1;i<s.y_km.length-1;i+=step)for(let j=1;j<s.x_km.length-1;j+=step){
      const x=s.d_dx[i][j],y=s.d_dy[i][j];if(!finite(x)||!finite(y))continue;
      const m=Math.hypot(x,y);if(m<1e-8)continue;
      const startX=s.x_km[j],startY=s.y_km[i],len=gridspacing;
      const endX=startX+len*x/m,endY=startY+len*y/m;
      arrowsX.push(startX,endX,null);arrowsY.push(startY,endY,null);
      const hx=endX-len*.23*x/m,hy=endY-len*.23*y/m;
      const side=len*.13;
      arrowsX.push(hx-side*y/m,endX,hx+side*y/m,null);arrowsY.push(hy+side*x/m,endY,hy-side*x/m,null);
    }
    plot('gradient-plot',[heat,{type:'scatter',mode:'lines',x:arrowsX,y:arrowsY,line:{color:'#273d43',width:1},hoverinfo:'skip',showlegend:false}],{xaxis:{title:'Easting (km)'},yaxis:{title:'Northing (km)',scaleanchor:'x',scaleratio:1},margin:{l:62,r:30,t:15,b:55}});
  }
  function showCharts(page){
    if(page==='overview' && data.report)drawOverview();
    if(page==='geography' && (data.points||data.surface))drawGeography();
    if(page==='calculus' && (data.geo||data.surface))drawCalculus();
  }
  async function prediction(event){
    event.preventDefault();
    const form=$('property-form'),button=$('predict-button'),error=$('predict-error');error.hidden=true;
    const inputs={};
    for(const [key,value] of new FormData(form).entries()){
      const node=form.elements[key];
      inputs[key]=node.type==='number'?(value.trim()===''?null:Number(value)):value.trim();
    }
    if(inputs.year_built>inputs.sale_year){error.textContent='Construction year cannot exceed the sale year.';error.hidden=false;return;}
    button.disabled=true;button.textContent='Calculating…';
    try{
      const response=await fetch('/api/official/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(inputs)});
      if(!response.ok){const body=await response.json();throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||`HTTP ${response.status}`));}
      const result=await response.json();
      mark('prediction-value',money(result.predicted_sale_price_usd));
      mark('prediction-caption','Model prediction for the specified historical property characteristics.');
      const s=result.spatial_evidence;mark('support-level',s?`${s.level} local data support`:'Evidence unavailable');
      const facts=$('support-facts');facts.replaceChildren();
      const items=s?[['10th-nearest development record',`${number(s.tenth_neighbor_km,2)} km`],['2024 comparable support cases',number(s.validation_reference_count)],['2024 support-group MAE',money(s.validation_group_mae_usd)],['2024 80th percentile absolute error',money(s.validation_group_p80_absolute_error_usd)]]:[];
      for(const [label,value] of items){const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;row.append(dt,dd);facts.append(row);}
    }catch(err){error.textContent=err.message;error.hidden=false;}
    finally{button.disabled=false;button.textContent='Calculate estimate →';}
  }
  function supportedPoints(){
    const s=data.surface;if(!s)return [];
    const good=[];
    for(let i=0;i<s.values.length;i++)for(let j=0;j<s.values[i].length;j++)if(finite(s.values[i][j])&&finite(s.d_dx[i][j])&&finite(s.d_dy[i][j]))good.push({lat:s.latitude[i][j],lon:s.longitude[i][j]});
    return good;
  }
  function initSamples(){
    const points=supportedPoints(),select=$('gradient-samples');select.replaceChildren(new Option('Select a supported location',''));
    if(!points.length)return;
    for(let k=1;k<=7;k++){
      const index=Math.min(points.length-1,Math.floor(points.length*k/8));
      const p=points[index];select.add(new Option(`Sample ${k} · ${p.lat.toFixed(3)}, ${p.lon.toFixed(3)}`,`${p.lat},${p.lon}`));
    }
  }
  $('gradient-samples').addEventListener('change',event=>{
    if(!event.target.value)return;
    const [lat,lon]=event.target.value.split(',');
    const f=$('gradient-form');f.elements.latitude.value=lat;f.elements.longitude.value=lon;
  });
  $('bearing-input').addEventListener('input',event=>mark('bearing-text',`${event.target.value}° clockwise from UTM grid north`));
  $('snap-point').addEventListener('click',()=>{
    const points=supportedPoints();if(!points.length)return;
    const f=$('gradient-form'),lat=Number(f.elements.latitude.value),lon=Number(f.elements.longitude.value);
    let closest=points[0],best=Infinity;
    for(const p of points){const d=Math.hypot((p.lat-lat)*111.2,(p.lon-lon)*85);if(d<best){closest=p;best=d;}}
    f.elements.latitude.value=closest.lat.toFixed(6);f.elements.longitude.value=closest.lon.toFixed(6);
    const error=$('gradient-error');error.textContent='Moved to nearest supported sample location.';error.hidden=false;
  });
  async function gradient(event){
    event.preventDefault();const form=$('gradient-form'),button=$('gradient-button'),error=$('gradient-error');error.hidden=true;button.disabled=true;button.textContent='Calculating…';
    const record={latitude:Number(form.elements.latitude.value),longitude:Number(form.elements.longitude.value),bearing:Number(form.elements.bearing.value)};
    try{
      const response=await fetch('/api/official/gradient',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(record)});
      if(!response.ok){const body=await response.json();throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||response.status));}
      const result=await response.json();
      $('derivative-result').hidden=false;
      mark('derivative-value',`${number(result.directional_derivative_usd_per_km)} USD/km`);
      const parent=$('derivative-facts');parent.replaceChildren();
      for(const [label,value] of [['Magnitude',`${number(result.gradient_magnitude_usd_per_km)} USD/km`],['Steepest-ascent grid bearing',`${number(result.steepest_ascent_grid_bearing_degrees,1)}°`],['Nearest supported sample',`${number(result.distance_to_gradient_km,2)} km`]]){
        const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;row.append(dt,dd);parent.append(row);
      }
    }catch(err){error.textContent=err.message;error.hidden=false;}
    finally{button.disabled=false;button.textContent='Calculate derivative';}
  }
  $('property-form').addEventListener('submit',prediction);
  $('gradient-form').addEventListener('submit',gradient);
  document.querySelectorAll('[data-layer]').forEach(btn=>btn.addEventListener('click',()=>{
    layer=btn.dataset.layer;
    document.querySelectorAll('[data-layer]').forEach(b=>{b.classList.toggle('active',b===btn);b.setAttribute('aria-pressed',String(b===btn));});
    if(data.points)drawMap();
  }));
  async function start(){
    const health=await getJSON('/api/official/health').catch(()=>null);
    const status=$('health');status.classList.add(health?.model_available?'ready':'not-ready');mark('health-text',health?.model_available?'Local model ready':'Model not available');
    const targets=[['report','/api/official/report'],['geo','/api/official/geo-report'],['reliability','/api/official/reliability-report'],['points','/official/data/points.json'],['surface','/official/data/surface.json']];
    await Promise.all(targets.map(async([key,url])=>{try{data[key]=await getJSON(url);}catch(_){data[key]=null;}}));
    if(data.report){const t=data.report;mark('model-name',`${t.model_selected.replaceAll('_',' ')} · 2025 held-out evaluation`);}
    initSamples();
    setPage(window.location.hash.slice(1)||'overview');
    if(!data.report&&$('page-overview').classList.contains('active'))drawOverview();
  }
  start().catch(err=>{mark('health-text','Local service unavailable');console.error('GeoValue startup:',err);});
})();
