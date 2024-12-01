import{a as k,b as w,e as H,f as S,g as T,h as A,i as I,j as N,k as E}from"./chunk-HDSZ7UIJ.js";import{Aa as v,Ba as M,Ca as r,Da as o,Ea as s,Fa as O,Ga as m,Ha as C,Ia as h,Ja as P,Ka as l,Ra as x,V as c,aa as _,ba as b,ca as y,ra as a,sa as p,za as f}from"./chunk-BDQAK3WA.js";var R=[{path:"",redirectTo:"home",pathMatch:"full"},{path:"home",loadComponent:()=>import("./chunk-7B4GQKXB.js").then(e=>e.HomeComponent)},{path:"topicsOverview/:categoryId",loadComponent:()=>import("./chunk-UZE3DSLZ.js").then(e=>e.TopicsOverviewComponent)},{path:"unitsOverview/:topicId",loadComponent:()=>import("./chunk-G4WEJJD5.js").then(e=>e.UnitsOverviewComponent)},{path:"unitContent/:unitId",loadComponent:()=>import("./chunk-G26B5P24.js").then(e=>e.UnitContentComponent)}];var z={providers:[H(),x({eventCoalescing:!0}),N(R)]};function L(e,n){if(e&1){let t=O();r(0,"li",10)(1,"button",11),m("click",function(){let g=_(t).$implicit,D=C(2);return b(D.openTopics(g.id))}),h(2),o()()}if(e&2){let t=n.$implicit;a(),M("color",t.color),a(),P(t.title)}}function j(e,n){if(e&1&&(r(0,"div",7)(1,"ul",8),f(2,L,3,3,"li",9),o()()),e&2){let t=C();a(2),v("ngForOf",t.categories)}}var d=class e{constructor(n,t,i){this.dataService=n;this.route=t;this.router=i}categories=[];isNavBarActive=!1;ngOnInit(){this.categories=this.dataService.getCategories()}openNavBar(){if(this.isNavBarActive=!this.isNavBarActive,this.isNavBarActive){let n=document.querySelector("body");n.style.overflow="hidden"}else{let n=document.querySelector("body");n.style.overflow="auto"}}openTopics(n){this.openNavBar(),this.router.navigate([`topicsOverview/${n}`]);let t=document.querySelector("input");t.checked=!1}openHome(){this.router.navigate(["home"])}static \u0275fac=function(t){return new(t||e)(p(E),p(T),p(I))};static \u0275cmp=c({type:e,selectors:[["app-header"]],standalone:!0,features:[l],decls:9,vars:1,consts:[[1,"home",3,"click"],[1,"hamburger"],["type","checkbox",3,"click"],["viewBox","0 0 32 32"],["d","M27 10 13 10C10.8 10 9 8.2 9 6 9 3.5 10.8 2 13 2 15.2 2 17 3.8 17 6L17 26C17 28.2 18.8 30 21 30 23.2 30 25 28.2 25 26 25 23.8 23.2 22 21 22L7 22",1,"line","line-top-bottom"],["d","M7 16 27 16",1,"line"],["class","nav-bar",4,"ngIf"],[1,"nav-bar"],[1,"categories"],["class","category",4,"ngFor","ngForOf"],[1,"category"],[3,"click"]],template:function(t,i){t&1&&(r(0,"header")(1,"button",0),m("click",function(){return i.openHome()}),h(2,"CodeSpace"),o(),r(3,"label",1)(4,"input",2),m("click",function(){return i.openNavBar()}),o(),y(),r(5,"svg",3),s(6,"path",4)(7,"path",5),o()(),f(8,j,3,1,"div",6),o()),t&2&&(a(8),v("ngIf",i.isNavBarActive))},dependencies:[w,k],styles:["header[_ngcontent-%COMP%]{width:100%;display:flex;align-items:center;justify-content:space-between;padding:10px 20px}.home[_ngcontent-%COMP%]{font-size:22px;padding:5px 10px;border:none;font-family:Cantarell-Regular;background-color:transparent}.hamburger[_ngcontent-%COMP%]{cursor:pointer;z-index:2}.hamburger[_ngcontent-%COMP%]   input[_ngcontent-%COMP%]{display:none}.hamburger[_ngcontent-%COMP%]   svg[_ngcontent-%COMP%]{height:2.5em;transition:transform .6s cubic-bezier(.4,0,.2,1)}.line[_ngcontent-%COMP%]{fill:none;stroke:#000;stroke-linecap:round;stroke-linejoin:round;stroke-width:2;transition:stroke-dasharray .6s cubic-bezier(.4,0,.2,1),stroke-dashoffset .6s cubic-bezier(.4,0,.2,1)}.line-top-bottom[_ngcontent-%COMP%]{stroke-dasharray:12 63}.hamburger[_ngcontent-%COMP%]   input[_ngcontent-%COMP%]:checked + svg[_ngcontent-%COMP%]{transform:rotate(-45deg)}.hamburger[_ngcontent-%COMP%]   input[_ngcontent-%COMP%]:checked + svg[_ngcontent-%COMP%]   .line-top-bottom[_ngcontent-%COMP%]{stroke-dasharray:20 300;stroke-dashoffset:-32.42}.nav-bar[_ngcontent-%COMP%]{width:100%;height:100vh;position:absolute;top:0;left:0;padding:20px;background-color:#fff;overflow-y:auto}.categories[_ngcontent-%COMP%]{width:100%;height:100%;display:flex;flex-direction:column;list-style:none}.category[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]{width:100%;font-size:3.4vw;text-align:start;border:none;background-color:transparent;transition:all .3s cubic-bezier(.165,.84,.44,1)}.category[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]:hover{font-size:4vw}@media (max-width: 900px){.category[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]{font-size:4vw}.category[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]:hover{font-size:4.6vw}}"]})};var u=class e{title="CodeSpace";static \u0275fac=function(t){return new(t||e)};static \u0275cmp=c({type:e,selectors:[["app-root"]],standalone:!0,features:[l],decls:2,vars:0,template:function(t,i){t&1&&s(0,"app-header")(1,"router-outlet")},dependencies:[A,d]})};S(u,z).catch(e=>console.error(e));