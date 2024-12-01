import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'home',
    pathMatch: 'full',
  },
  {
    path: 'home',
    loadComponent: () => import('./components/home/home.component').then(m => m.HomeComponent),
  },
  {
    path: 'topicsOverview/:categoryId',
    loadComponent: () => import('./components/topics-overview/topics-overview.component').then(m => m.TopicsOverviewComponent),
  },
  {
    path: 'unitsOverview/:topicId',
    loadComponent: () => import('./components/units-overview/units-overview.component').then(m => m.UnitsOverviewComponent),
  },
  {
    path: 'unitContent/:unitId',
    loadComponent: () => import('./components/unit-content/unit-content.component').then(m => m.UnitContentComponent),
  }
];
