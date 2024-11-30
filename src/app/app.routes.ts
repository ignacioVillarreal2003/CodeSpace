import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'topicsOverview',
    pathMatch: 'full',
  },
  {
    path: 'topicsOverview',
    loadComponent: () => import('./components/topics-overview/topics-overview.component').then(m => m.TopicsOverviewComponent),
  },
  {
    path: 'unitsOverview/:topicId',
    loadComponent: () => import('./components/units-overview/units-overview.component').then(m => m.UnitsOverviewComponent),
  },
  {
    path: 'unitContent/:unitId',
    loadComponent: () => import('./components/unit-content/unit-content.component').then(m => m.UnitContentComponent),
  },
];
