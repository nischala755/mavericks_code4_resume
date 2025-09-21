# Dashboard and Analytics continuation for app.py

    with tab3:
        st.header("📊 Evaluation Dashboard")
        
        # Job selection for dashboard
        jobs = system.get_job_descriptions()
        if jobs:
            dashboard_col1, dashboard_col2 = st.columns([3, 1])
            
            with dashboard_col1:
                job_options = {f"{job['title']} - {job['company']}": job['id'] for job in jobs}
                selected_job_display = st.selectbox(
                    "Select Job for Dashboard", 
                    options=list(job_options.keys()), 
                    key="dashboard_job"
                )
                selected_job_id = job_options[selected_job_display]
            
            with dashboard_col2:
                auto_refresh = st.checkbox("Auto Refresh", value=False)
                if auto_refresh:
                    st.rerun()
            
            evaluations = system.get_evaluations_for_job(selected_job_id)
            
            if evaluations:
                # Enhanced summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Applications", len(evaluations))
                
                with col2:
                    high_count = len([e for e in evaluations if e['verdict'] == 'High'])
                    st.metric("High Suitability", high_count, delta=f"{(high_count/len(evaluations)*100):.1f}%")
                
                with col3:
                    medium_count = len([e for e in evaluations if e['verdict'] == 'Medium'])
                    st.metric("Medium Suitability", medium_count, delta=f"{(medium_count/len(evaluations)*100):.1f}%")
                
                with col4:
                    avg_score = sum([e['relevance_score'] for e in evaluations]) / len(evaluations)
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                with col5:
                    top_score = max([e['relevance_score'] for e in evaluations])
                    st.metric("Top Score", f"{top_score}%")
                
                # Interactive charts
                st.subheader("📈 Score Distribution")
                
                # Create score distribution chart
                scores = [e['relevance_score'] for e in evaluations]
                score_ranges = {
                    '90-100': len([s for s in scores if 90 <= s <= 100]),
                    '80-89': len([s for s in scores if 80 <= s < 90]),
                    '70-79': len([s for s in scores if 70 <= s < 80]),
                    '60-69': len([s for s in scores if 60 <= s < 70]),
                    '50-59': len([s for s in scores if 50 <= s < 60]),
                    'Below 50': len([s for s in scores if s < 50])
                }
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Score distribution bar chart
                    range_df = pd.DataFrame(list(score_ranges.items()), columns=['Score Range', 'Count'])
                    st.bar_chart(range_df.set_index('Score Range'))
                
                with chart_col2:
                    # Verdict pie chart data
                    verdict_counts = {}
                    for evaluation in evaluations:
                        verdict = evaluation['verdict']
                        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                    
                    verdict_df = pd.DataFrame(list(verdict_counts.items()), columns=['Verdict', 'Count'])
                    st.write("**Verdict Distribution**")
                    for _, row in verdict_df.iterrows():
                        percentage = (row['Count'] / len(evaluations)) * 100
                        st.write(f"**{row['Verdict']}:** {row['Count']} ({percentage:.1f}%)")
                
                # Top candidates table
                st.subheader("🏆 Top Candidates")
                
                # Filter and display options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    min_score = st.slider("Minimum Score", 0, 100, 60)
                
                with filter_col2:
                    verdict_filter = st.multiselect(
                        "Filter by Verdict", 
                        ['High', 'Medium', 'Low'],
                        default=['High', 'Medium']
                    )
                
                with filter_col3:
                    show_count = st.selectbox("Show Top", [10, 20, 50, 100], index=0)
                
                # Apply filters
                filtered_evaluations = [
                    e for e in evaluations 
                    if e['relevance_score'] >= min_score and e['verdict'] in verdict_filter
                ]
                
                # Display filtered results
                if filtered_evaluations:
                    display_evaluations = filtered_evaluations[:show_count]
                    
                    # Create enhanced table
                    table_data = []
                    for i, eval in enumerate(display_evaluations, 1):
                        table_data.append({
                            'Rank': i,
                            'Candidate': eval['candidate_name'],
                            'Score': eval['relevance_score'],
                            'Verdict': eval['verdict'],
                            'File': eval.get('file_name', 'N/A'),
                            'Date': eval['created_at'],
                            'Matching Skills': len(eval.get('matching_skills', [])),
                            'Missing Skills': len(eval.get('missing_skills', []))
                        })
                    
                    table_df = pd.DataFrame(table_data)
                    
                    # Style the table
                    def highlight_rows(row):
                        if row['Verdict'] == 'High':
                            return ['background-color: #d4edda'] * len(row)
                        elif row['Verdict'] == 'Medium':
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)
                    
                    styled_table = table_df.style.apply(highlight_rows, axis=1)
                    st.dataframe(styled_table, use_container_width=True)
                    
                    # Quick actions
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("📧 Export for HR", type="secondary"):
                            # Create detailed export
                            export_data = []
                            for eval in display_evaluations:
                                export_data.append({
                                    'Candidate_Name': eval['candidate_name'],
                                    'Relevance_Score': eval['relevance_score'],
                                    'Verdict': eval['verdict'],
                                    'Matching_Skills': ', '.join(eval.get('matching_skills', [])),
                                    'Missing_Skills': ', '.join(eval.get('missing_skills', [])),
                                    'Suggestions': ', '.join(eval.get('suggestions', [])),
                                    'File_Name': eval.get('file_name', ''),
                                    'Evaluation_Date': eval['created_at']
                                })
                            
                            export_df = pd.DataFrame(export_data)
                            csv_export = export_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download HR Report",
                                data=csv_export,
                                file_name=f"hr_report_{selected_job_display.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    with action_col2:
                        if st.button("⭐ Shortlist Top 5", type="secondary"):
                            top_5 = display_evaluations[:5]
                            shortlist_data = pd.DataFrame([
                                {
                                    'Candidate': e['candidate_name'],
                                    'Score': e['relevance_score'],
                                    'Verdict': e['verdict'],
                                    'Contact_Ready': 'Yes' if e['relevance_score'] >= 70 else 'Review Required'
                                }
                                for e in top_5
                            ])
                            
                            csv_shortlist = shortlist_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Top 5",
                                data=csv_shortlist,
                                file_name=f"top_5_candidates_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    with action_col3:
                        if st.button("📊 Detailed Analysis", type="secondary"):
                            st.info("Feature coming soon: Detailed skill gap analysis and recommendations")
                
                else:
                    st.info("No candidates match the selected filters.")
                
                # Recent activity feed
                st.subheader("📅 Recent Activity")
                recent_evaluations = evaluations[:5]
                
                for eval in recent_evaluations:
                    with st.expander(f"📄 {eval['candidate_name']} - {eval['relevance_score']}% ({eval['verdict']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**File:** {eval.get('file_name', 'N/A')}")
                            st.write(f"**Date:** {eval['created_at']}")
                            st.write(f"**Score:** {eval['relevance_score']}%")
                        
                        with col2:
                            if eval.get('matching_skills'):
                                st.write(f"**Matching Skills:** {len(eval['matching_skills'])}")
                                with st.expander("View Skills"):
                                    for skill in eval['matching_skills'][:5]:
                                        st.write(f"✅ {skill}")
                                    if len(eval['matching_skills']) > 5:
                                        st.write(f"... and {len(eval['matching_skills']) - 5} more")
            
            else:
                st.info("📝 No evaluations found for this job position. Start evaluating resumes to see data here!")
                
                if st.button("➕ Evaluate Resumes Now", type="primary"):
                    st.switch_page("Resume Evaluation")
        
        else:
            st.warning("📋 No job descriptions available. Please add a job description first.")

    with tab4:
        st.header("📈 Advanced Analytics & Insights")
        
        # Get comprehensive analytics data
        analytics_data = system.get_analytics_data()
        
        if analytics_data['total_evaluations'] > 0:
            # System overview
            st.subheader("🎯 System Overview")
            
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            
            with overview_col1:
                st.metric(
                    "Total Jobs Posted", 
                    analytics_data['total_jobs'],
                    help="Total number of job descriptions in the system"
                )
            
            with overview_col2:
                st.metric(
                    "Total Applications", 
                    analytics_data['total_evaluations'],
                    help="Total number of resumes evaluated"
                )
            
            with overview_col3:
                st.metric(
                    "System Average Score", 
                    f"{analytics_data['avg_score']:.1f}%",
                    help="Average relevance score across all evaluations"
                )
            
            with overview_col4:
                if analytics_data['total_jobs'] > 0:
                    avg_applications = analytics_data['total_evaluations'] / analytics_data['total_jobs']
                    st.metric(
                        "Avg Applications/Job", 
                        f"{avg_applications:.1f}",
                        help="Average number of applications per job posting"
                    )
            
            # Performance trends
            st.subheader("📊 Performance Analytics")
            
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.write("**📈 Score Distribution Analysis**")
                
                scores = analytics_data['all_scores']
                if scores:
                    # Calculate percentiles
                    import numpy as np
                    percentiles = {
                        '90th Percentile': np.percentile(scores, 90),
                        '75th Percentile': np.percentile(scores, 75),
                        '50th Percentile (Median)': np.percentile(scores, 50),
                        '25th Percentile': np.percentile(scores, 25)
                    }
                    
                    for label, value in percentiles.items():
                        st.write(f"**{label}:** {value:.1f}%")
                    
                    # Score histogram
                    hist_data = pd.DataFrame({'Scores': scores})
                    st.bar_chart(hist_data['Scores'].value_counts().sort_index())
            
            with trend_col2:
                st.write("**🎯 Verdict Analysis**")
                
                verdict_dist = analytics_data['verdict_distribution']
                total_evals = analytics_data['total_evaluations']
                
                if verdict_dist:
                    for verdict, count in verdict_dist.items():
                        percentage = (count / total_evals) * 100
                        st.write(f"**{verdict}:** {count} ({percentage:.1f}%)")
                    
                    # Quality metrics
                    high_quality_rate = verdict_dist.get('High', 0) / total_evals * 100
                    acceptable_rate = (verdict_dist.get('High', 0) + verdict_dist.get('Medium', 0)) / total_evals * 100
                    
                    st.write("---")
                    st.write(f"**High Quality Rate:** {high_quality_rate:.1f}%")
                    st.write(f"**Acceptable Rate:** {acceptable_rate:.1f}%")
            
            # Top performers
            st.subheader("🏆 Top Performing Candidates")
            
            top_candidates = analytics_data['top_candidates']
            if top_candidates:
                top_df = pd.DataFrame(top_candidates)
                
                # Display top candidates table
                st.dataframe(
                    top_df[['name', 'score', 'verdict', 'date']].rename(columns={
                        'name': 'Candidate Name',
                        'score': 'Score (%)',
                        'verdict': 'Verdict',
                        'date': 'Evaluation Date'
                    }),
                    use_container_width=True
                )
                
                # Top performer insights
                if len(top_candidates) >= 5:
                    top_5_avg = sum(c['score'] for c in top_candidates[:5]) / 5
                    st.info(f"💡 **Insight:** Top 5 candidates average {top_5_avg:.1f}% relevance score")
            
            # Skill gap analysis (future enhancement placeholder)
            st.subheader("🔍 Skill Gap Analysis")
            
            with st.expander("📋 Most In-Demand Skills (Coming Soon)"):
                st.info("This feature will analyze the most frequently required skills across all job postings and identify common skill gaps in applications.")
            
            with st.expander("📈 Market Trends (Coming Soon)"):
                st.info("This section will show trending skills, emerging requirements, and candidate pool insights over time.")
            
            # Export comprehensive report
            st.subheader("📥 Export Analytics")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📊 Generate System Report", type="primary"):
                    # Create comprehensive system report
                    report_data = {
                        'Report_Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Total_Jobs': analytics_data['total_jobs'],
                        'Total_Evaluations': analytics_data['total_evaluations'],
                        'Average_Score': analytics_data['avg_score'],
                        'High_Quality_Candidates': analytics_data['verdict_distribution'].get('High', 0),
                        'Medium_Quality_Candidates': analytics_data['verdict_distribution'].get('Medium', 0),
                        'Low_Quality_Candidates': analytics_data['verdict_distribution'].get('Low', 0)
                    }
                    
                    # Add top candidates to report
                    for i, candidate in enumerate(analytics_data['top_candidates'][:10], 1):
                        report_data[f'Top_{i}_Candidate'] = candidate['name']
                        report_data[f'Top_{i}_Score'] = candidate['score']
                    
                    report_df = pd.DataFrame([report_data])
                    csv_report = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download System Report",
                        data=csv_report,
                        file_name=f"system_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with export_col2:
                if st.button("📋 Export All Data", type="secondary"):
                    st.info("Preparing comprehensive data export... This feature will export all job descriptions, evaluations, and analytics in a structured format.")
        
        else:
            st.info("📊 No analytics data available yet. Start by adding job descriptions and evaluating resumes to see insights here!")
            
            # Getting started guide
            with st.expander("🚀 Getting Started Guide"):
                st.markdown("""
                To see analytics and insights:
                
                1. **Add Job Descriptions** - Go to the Job Management tab and add at least one job posting
                2. **Evaluate Resumes** - Upload and evaluate candidate resumes against your job requirements
                3. **Review Results** - Check the Dashboard for individual job insights
                4. **Analyze Trends** - Return here to see system-wide analytics and performance metrics
                
                The more data you add, the more valuable insights you'll get!
                """)

    with tab5:
        st.header("⚙️ System Settings & Configuration")
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        with st.expander("Gemini AI Settings"):
            current_key_status = "✅ Connected" if api_key else "❌ Not Connected"
            st.write(f"**API Status:** {current_key_status}")
            
            if st.button("🔄 Refresh API Connection"):
                st.cache_data.clear()
                st.rerun()
            
            st.info("💡 To change your API key, refresh the page and enter a new key in the sidebar.")
        
        # Database Management
        st.subheader("🗄️ Database Management")
        
        with st.expander("Database Information"):
            try:
                cursor = db_conn.cursor()
                
                # Get table sizes
                cursor.execute("SELECT COUNT(*) FROM job_descriptions")
                job_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM resume_evaluations")
                eval_count = cursor.fetchone()[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Job Descriptions", job_count)
                with col2:
                    st.metric("Resume Evaluations", eval_count)
                
                # Database file size
                db_path = Path('data/resume_relevance.db')
                if db_path.exists():
                    db_size = db_path.stat().st_size / 1024 / 1024  # MB
                    st.write(f"**Database Size:** {db_size:.2f} MB")
                
            except Exception as e:
                st.error(f"Database error: {str(e)}")
        
        # Performance Settings
        st.subheader("⚡ Performance Settings")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write("**Cache Management**")
            
            if st.button("🧹 Clear All Cache"):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
            
            # Show cache statistics
            cache_stats = st.cache_data.get_stats()
            st.write(f"**Cached Functions:** {len(cache_stats)}")
        
        with perf_col2:
            st.write("**Processing Limits**")
            
            st.write(f"**Max File Size:** {settings['max_file_size']} MB")
            st.write(f"**Caching Enabled:** {'Yes' if settings['enable_caching'] else 'No'}")
            st.write(f"**Debug Mode:** {'Yes' if settings['show_debug_info'] else 'No'}")
        
        # Data Export/Import
        st.subheader("📁 Data Management")
        
        data_col1, data_col2 = st.columns(2)
        
        with data_col1:
            st.write("**Export Data**")
            
            if st.button("📥 Export All Job Descriptions"):
                try:
                    jobs = system.get_job_descriptions()
                    if jobs:
                        jobs_df = pd.DataFrame(jobs)
                        csv_data = jobs_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Jobs CSV",
                            data=csv_data,
                            file_name=f"job_descriptions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No job descriptions to export")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with data_col2:
            st.write("**Backup & Restore**")
            
            if st.button("💾 Create Backup"):
                st.info("Backup functionality coming soon. Currently, your data is stored in the 'data' folder.")
            
            st.file_uploader(
                "Restore from Backup",
                type=['db', 'sqlite'],
                help="Upload a database backup file (coming soon)"
            )
        
        # System Information
        st.subheader("ℹ️ System Information")
        
        with st.expander("Technical Details"):
            st.write("**Application Version:** 2.0.0")
            st.write("**Streamlit Version:**", st.__version__)
            st.write("**Database Type:** SQLite")
            st.write("**AI Model:** Google Gemini Pro")
            
            # Dependency versions
            st.write("**Dependencies:**")
            for dep_name, available in DEPENDENCIES_AVAILABLE.items():
                status = "✅ Installed" if available else "❌ Missing"
                st.write(f"- {dep_name}: {status}")
        
        # Help & Support
        st.subheader("❓ Help & Support")
        
        with st.expander("Quick Help"):
            st.markdown("""
            **Common Issues:**
            
            1. **API Key Problems:** Ensure your Gemini API key is valid and has quota remaining
            2. **File Upload Issues:** Check file format (PDF/DOCX only) and size limits
            3. **Slow Processing:** Large files or many API calls may take time
            4. **Database Errors:** Try clearing cache or restarting the application
            
            **Best Practices:**
            
            - Use clear, detailed job descriptions for better matching
            - Ensure resume files are text-readable (not scanned images)
            - Process resumes in smaller batches for better performance
            - Regularly backup your data
            """)
        
        with st.expander("Feature Requests & Feedback"):
            st.markdown("""
            **Planned Features:**
            
            - 📊 Advanced skill gap analytics
            - 🔄 Integration with HR systems
            - 📧 Email notifications for high-match candidates
            - 🎨 Custom scoring weight configuration
            - 📈 Historical trend analysis
            - 🔍 AI-powered candidate recommendations
            
            **Contact:** For support or feature requests, please contact the development team.
            """)

if __name__ == "__main__":
    main()