import{a as b}from"./chunk-JVDPG3LW.js";import{A as x,C,G as O,I as y,f as a,g as l,h as d,k as o,l as n,m,n as p,o as r,p as s,r as v,s as u,t as f,v as g,w,x as _,y as h}from"./chunk-PHHWFVGT.js";var P=i=>({"background-color":i});function I(i,e){if(i&1){let t=v();r(0,"li",3)(1,"button",4),u("click",function(){let M=l(t).$implicit,S=f();return d(S.openTopic(M.id))}),r(2,"div",5)(3,"h2"),g(4),s()()()()}if(i&2){let t=e.$implicit;o(2),p("ngStyle",h(2,P,t.color)),o(2),w(t.title)}}var T=class i{constructor(e,t,c){this.dataService=e;this.route=t;this.router=c}topics=[];ngOnInit(){let e=this.route.snapshot.paramMap.get("categoryId");e!=null&&(this.topics=this.dataService.getTopics(e))}openTopic(e){this.router.navigate([`unitsOverview/${e}`])}static \u0275fac=function(t){return new(t||i)(n(b),n(O),n(y))};static \u0275cmp=a({type:i,selectors:[["app-topics-overview"]],standalone:!0,features:[_],decls:3,vars:1,consts:[[1,"topics-overview"],[1,"topics"],["class","topic",4,"ngFor","ngForOf"],[1,"topic"],[3,"click"],[1,"title",3,"ngStyle"]],template:function(t,c){t&1&&(r(0,"div",0)(1,"ul",1),m(2,I,5,4,"li",2),s()()),t&2&&(o(2),p("ngForOf",c.topics))},dependencies:[x,C],styles:[".topics-overview[_ngcontent-%COMP%]{width:100%;height:100%;display:flex;align-items:center;justify-content:center}.topics[_ngcontent-%COMP%]{width:90%;display:grid;grid-template-columns:repeat(4,1fr);gap:24px;list-style:none}button[_ngcontent-%COMP%]{width:100%;aspect-ratio:1;border:none;background-color:transparent;display:flex;flex-direction:column}.title[_ngcontent-%COMP%]{width:100%;height:100%;display:flex;align-items:center;justify-content:center;background-color:#000;transition:box-shadow .3s;box-shadow:inset 0 0 0 1px #fff}button[_ngcontent-%COMP%]:hover   .title[_ngcontent-%COMP%]{box-shadow:inset 0 0 0 1vw #fff}h2[_ngcontent-%COMP%]{font-size:18px}@media (max-width: 900px){.topics[_ngcontent-%COMP%]{grid-template-columns:repeat(3,1fr)}h2[_ngcontent-%COMP%]{font-size:12px}}"]})};export{T as TopicsOverviewComponent};
