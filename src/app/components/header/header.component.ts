import { Component } from '@angular/core';
import { NgIf, NgFor } from '@angular/common';
import { Category } from '../../content/category';
import { DataService } from '../../services/data.service';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [NgIf, NgFor],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {
  categories: Category[] = []
  isNavBarActive: boolean = false;

  constructor(private dataService: DataService, private route: ActivatedRoute, private router: Router) {}

  ngOnInit(): void {
    this.categories = this.dataService.getCategories();
  }

  openNavBar(): void {
    this.isNavBarActive = !this.isNavBarActive;
  }

  openTopics(categoryId: string): void {
    this.openNavBar();
    this.router.navigate([`topicsOverview/${categoryId}`]);
  }
}
