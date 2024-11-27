import { Component } from '@angular/core';
import {NgForOf, NgStyle} from "@angular/common";
import {Category} from '../../content/category';
import {DataService} from '../../services/data.service';
import {Router} from '@angular/router';

@Component({
  selector: 'app-categories-overview',
  standalone: true,
  imports: [
    NgForOf,
    NgStyle
  ],
  templateUrl: './categories-overview.component.html',
  styleUrl: './categories-overview.component.css'
})
export class CategoriesOverviewComponent {
  categories: Category[] = []

  constructor(private dataService: DataService, private router: Router) {}

  ngOnInit(): void {
    this.categories = this.dataService.getCategories();
  }

  openTopics(categoryId: string): void {
    console.log(categoryId)
    this.router.navigate([`topicsOverview/${categoryId}`])
  }
}
