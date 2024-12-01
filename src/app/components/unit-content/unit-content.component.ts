import { Component  } from '@angular/core';
import { NgFor, NgIf } from "@angular/common";
import hljs from 'highlight.js';
import { ActivatedRoute } from '@angular/router';
import { DataService } from '../../services/data.service';
import { HttpClient } from '@angular/common/http';
import python from 'highlight.js/lib/languages/python';
import java from 'highlight.js/lib/languages/java';
import csharp from 'highlight.js/lib/languages/csharp';
import { marked } from 'marked';
import {Unit} from '../../content/unit';

hljs.registerLanguage('python', python);
hljs.registerLanguage('java', java);
hljs.registerLanguage('csharp', csharp);

@Component({
  selector: 'app-unit-content',
  standalone: true,
  imports: [
    NgFor,
    NgIf
  ],
  templateUrl: './unit-content.component.html',
  styleUrls: ['./unit-content.component.css']
})
export class UnitContentComponent {

  unit: Unit | undefined = undefined;
  markdownContent: string = '';
  htmlContent: string = '';

  constructor(private route: ActivatedRoute, private dataService: DataService, private http: HttpClient) {}

  ngOnInit(): void {
    const unitId: string | null = this.route.snapshot.paramMap.get('unitId');
    if (unitId) {
      const unit: Unit | undefined = this.dataService.getUnit(unitId);
      if (unit) {
        this.unit = unit;
        const filePath: string = `assets/markdown/${this.unit.content}`;
        this.http.get(filePath, { responseType: 'text' }).subscribe({
          next: (content: string) => {
            this.markdownContent = content;
            this.htmlContent = marked(content) as string;
            this.applyHighlighting();
          },
          error: (err): void => {
            console.error('Error al cargar el archivo Markdown:', err);
          }
        });
      }
    }
  }

  private applyHighlighting(): void {
    setTimeout(() => {
      document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block as HTMLElement);
      });
    }, 10);
  }
}
