##Project Question##

**What is the question you hope to answer?**

Can we predict the "successful adoption" of a user of an enterprise collaboration and knowledge management platform based on the user's
on-platform activities, demographics (more in the sense of work-place demographics like general function, seniority, etc.), off-platform activities (e.g. training classes) and organizational support (adoption of direct manager and senior leadership within the user's organizational hierarchy)? The real objective is to develop a process to identify the key ingredients that are predictive of a user going from initial logon to value-creating user so these activities can be encouraged and environmental factors strengthened / optimized.

**What data are you planning to use to answer that question?**

Deciding on a good and defensible definition of "successful adoption" will be a key determinant in whether this project is actually useful and not just an academic exercise; this definition will serve as the proxy for whether the value of the of the platform is actually being realized. My intuition is that "unsuccessful" users either never really start using the system or end up fading away after an initial burst of activity and thus, at a high-level, I plan to split users into 3 buckets: joined but never really used, joined but stopped really using within 6-months, and joined and "using" after six months. The specific definitions of these three categories will need to be further refined but this is sufficient to understand what data we will need.

I believe the following data will be most useful (in order of usefulness):
1. On-platform "analytics" (specific activities of all users captured in detail)
2. Individual profile data including things like job title (function and seniority) place in the org structure, original join date, etc.
3. Off-platform activities like registrations/attendance at training, roadshows, other events; this is in a bunch of spreadsheets and will likely require some manual cleaning (this is an overdue task in general).
4. General organization information (org charts with senior leadership for divisions/functions) and maybe holiday schedules or other seasonal adjustments (only if there are significant seasonal effects obvious in the data)

**What do you know about the data so far?**

I've chosen a platform rollout which is right now getting to the point where there are enough users (~7,500) who have been in the system for 6-months or more that there should be a sufficient sample set to have significant numbers of both "successful" and "unsuccessful" adoption users. Additionally, the total target population is > 250,000 so any insights derived have the potential to meaningfully inform our strategy going forward and hopefully this model can be "productionalized" sufficiently to provide feedback on whether any adjustments we make are impacting results or if changes in demographics of the current rollout population would suggest a change in strategy.

There's a ton of data about individual user activity on-platform (after all it's a web app and capturing user activity is what these systems do); it's in a couple of SQL databases that have convoluted schemas but nothing terrible. Producing usable datasets out of this will require some SQL magic but nothing I'm too worried about other than making sure I don't turn this into a big data problem from a volume perspective. We do lots of reporting and visualization on the basic metrics on user acquisition but it is more focused on the initial step of getting the user to join the community (and how activities and circumstances are impacting this initial transaction) and less on metrics of users 6-months after initial login. Because this data is system generated it's generally pretty clean (we have come across some outlier users who do strange things in the system to optimize their performance metrics).

Individual Profile data is easily pulled from a SQL database. Unfortunately it is largely optional and user provided; thus tends to be spotty in both availability and quality.

Data for off-platform activities is in a bunch of spreadsheets. It has been generally reliable but will likely need some basic scrubbing (this clean-up is an overdue task in general and, for better or worse, the overall volume of these activities is significantly lower so this should be manageable).

General org information is a something that I'll probably have to put together by hand if it looks like it will be useful.

**Why did you choose this topic?**

I've dealt with building and rolling out enterprise software to large organizations for most of my career. My experience is that the success of transformational software projects is only minorly dependent of how good the software my team creates; the software just has to be good enough to not turn users off. Success is largely a function of the larger team's organizational change management plan (both in original quality and in execution) and the organization creating the right incentives and environment to get people to actually become active users of the software. The overall project's ability to get users to adopt the new software in a meaningful way that actual transforms the way they work is what matters most. There is no category of software for which this is more true than for enterprise knowledge management and collaboration software. Mandates from above do not result in the the sort of adoption that actually delivers any sort of real value. The teams I work with now intuitively understand this and put appropriate resources into driving adoption. However, all of the efforts I have seen to date are a largely a function of intuition and anecdotal evidence. I'd like to see if we can find a way to begin using data to focus our efforts on the ingredients that maximize our probability of creating successful users.
