import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'categoriesOverview',
    pathMatch: 'full',
  },
  {
    path: 'categoriesOverview',
    loadComponent: () => import('./components/categories-overview/categories-overview.component').then(m => m.CategoriesOverviewComponent),
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
  },
];
